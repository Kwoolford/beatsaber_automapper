"""Tests for NotePredictor model and NotePredictionLitModule."""

import pytest
import torch

from beatsaber_automapper.data.tokenizer import (
    ANGLE_OFFSET_OFFSET,
    BOS,
    COL_OFFSET,
    COLOR_OFFSET,
    DIR_OFFSET,
    EOS,
    NOTE,
    PAD,
    ROW_OFFSET,
    SEP,
    tokens_to_structured,
)
from beatsaber_automapper.models.note_predictor import (
    MAX_SLOTS,
    NotePredictor,
)


class TestTokensToStructured:
    """Test the tokens_to_structured adapter function."""

    def test_single_note(self):
        """Single NOTE event parsed correctly."""
        # NOTE(4) COLOR_RED(10) COL_1(13) ROW_0(16) DIR_DOWN(20) ANGLE_0(31) EOS(1)
        tokens = [NOTE, COLOR_OFFSET, COL_OFFSET + 1, ROW_OFFSET, DIR_OFFSET + 1,
                  ANGLE_OFFSET_OFFSET + 3, EOS]
        result = tokens_to_structured(tokens)
        assert result["n_notes"] == 1
        assert result["slots"][0]["color"] == 0  # red
        assert result["slots"][0]["col"] == 1
        assert result["slots"][0]["row"] == 0
        assert result["slots"][0]["direction"] == 1  # down
        assert result["slots"][0]["angle"] == 3  # center
        assert result["slots"][0]["event_type"] == 0  # note
        # Inactive slots
        assert result["slots"][1]["color"] == 2  # none
        assert result["slots"][2]["color"] == 2  # none

    def test_two_notes_with_sep(self):
        """Two notes separated by SEP."""
        tokens = [
            NOTE, COLOR_OFFSET, COL_OFFSET, ROW_OFFSET, DIR_OFFSET + 1,
            ANGLE_OFFSET_OFFSET + 3,
            SEP,
            NOTE, COLOR_OFFSET + 1, COL_OFFSET + 3, ROW_OFFSET + 1, DIR_OFFSET,
            ANGLE_OFFSET_OFFSET + 3,
            EOS,
        ]
        result = tokens_to_structured(tokens)
        assert result["n_notes"] == 2
        assert result["slots"][0]["color"] == 0  # red
        assert result["slots"][1]["color"] == 1  # blue
        assert result["slots"][1]["col"] == 3
        assert result["slots"][1]["row"] == 1

    def test_empty_sequence(self):
        """EOS-only sequence → 0 notes."""
        result = tokens_to_structured([EOS])
        assert result["n_notes"] == 0
        assert all(s["color"] == 2 for s in result["slots"])

    def test_pad_tokens_ignored(self):
        """PAD and BOS tokens are filtered out."""
        tokens = [BOS, NOTE, COLOR_OFFSET, COL_OFFSET, ROW_OFFSET,
                  DIR_OFFSET, ANGLE_OFFSET_OFFSET, EOS, PAD, PAD]
        result = tokens_to_structured(tokens)
        assert result["n_notes"] == 1

    def test_max_slots_capping(self):
        """More than max_slots events → only first max_slots kept."""
        note = [NOTE, COLOR_OFFSET, COL_OFFSET, ROW_OFFSET,
                DIR_OFFSET, ANGLE_OFFSET_OFFSET]
        tokens = note + [SEP] + note + [SEP] + note + [SEP] + note + [EOS]
        result = tokens_to_structured(tokens, max_slots=3)
        assert result["n_notes"] == 3
        assert len(result["slots"]) == 3


class TestNotePredictor:
    """Test NotePredictor model forward pass."""

    @pytest.fixture
    def model(self):
        return NotePredictor(d_model=64, nhead=4, num_pool_layers=1,
                             dim_feedforward=128, prev_context_k=0)

    def test_output_shapes(self, model):
        """All output heads have correct shapes."""
        bs, t, d = 4, 32, 64
        audio = torch.randn(bs, t, d)
        diff = torch.zeros(bs, dtype=torch.long)
        genre = torch.zeros(bs, dtype=torch.long)

        out = model(audio, diff, genre)

        assert out["n_notes"].shape == (bs, 4)
        assert out["color"].shape == (bs, MAX_SLOTS, 3)
        assert out["col"].shape == (bs, MAX_SLOTS, 4)
        assert out["row"].shape == (bs, MAX_SLOTS, 3)
        assert out["direction"].shape == (bs, MAX_SLOTS, 9)
        assert out["angle"].shape == (bs, MAX_SLOTS, 7)
        assert out["event_type"].shape == (bs, MAX_SLOTS, 5)

    def test_with_prev_context(self):
        """Model works with previous onset context."""
        model = NotePredictor(d_model=64, nhead=4, num_pool_layers=1,
                              dim_feedforward=128, prev_context_k=4)
        bs, t, d, k, s = 2, 32, 64, 4, 16
        audio = torch.randn(bs, t, d)
        diff = torch.zeros(bs, dtype=torch.long)
        genre = torch.zeros(bs, dtype=torch.long)
        prev = torch.randint(0, 167, (bs, k, s))

        out = model(audio, diff, genre, prev_tokens=prev)
        assert out["n_notes"].shape == (bs, 4)

    def test_gradients_flow(self):
        """Gradients flow through all output heads."""
        model = NotePredictor(d_model=64, nhead=4, num_pool_layers=1,
                              dim_feedforward=128)
        bs, t, d = 2, 16, 64
        audio = torch.randn(bs, t, d)
        diff = torch.zeros(bs, dtype=torch.long)
        genre = torch.zeros(bs, dtype=torch.long)

        out = model(audio, diff, genre)

        # Sum all outputs and backprop
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        # Check gradients exist on key parameters
        assert model.slot_queries.grad is not None
        assert model.slot_queries.grad.norm() > 0


class TestNotePredictionLitModule:
    """Test NotePredictionLitModule training/validation."""

    @pytest.fixture
    def module(self):
        from beatsaber_automapper.training.note_module import NotePredictionLitModule
        return NotePredictionLitModule(
            n_mels=48,
            encoder_d_model=64,
            encoder_nhead=4,
            encoder_num_layers=1,
            encoder_dim_feedforward=128,
            encoder_dropout=0.0,
            pred_nhead=4,
            pred_num_pool_layers=1,
            pred_dim_feedforward=128,
            pred_dropout=0.0,
            prev_context_k=0,
            warmup_steps=10,
        )

    def _make_batch(self, b=4, n_mels=48, ctx=32, seq_len=16):
        """Create a fake batch matching SequenceDataset format."""
        # Create token sequences with valid notes
        tokens = torch.full((b, seq_len), PAD, dtype=torch.long)
        for i in range(b):
            tokens[i, 0] = NOTE
            tokens[i, 1] = COLOR_OFFSET + (i % 2)
            tokens[i, 2] = COL_OFFSET + (i % 4)
            tokens[i, 3] = ROW_OFFSET + (i % 3)
            tokens[i, 4] = DIR_OFFSET + 1
            tokens[i, 5] = ANGLE_OFFSET_OFFSET + 3
            tokens[i, 6] = EOS

        return {
            "mel": torch.randn(b, n_mels, ctx),
            "tokens": tokens,
            "difficulty": torch.randint(0, 5, (b,)),
            "genre": torch.zeros(b, dtype=torch.long),
            "structure": torch.randn(b, 8, ctx),
        }

    def test_training_step_returns_scalar(self, module):
        """Training step returns a scalar loss."""
        batch = self._make_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_validation_step_runs(self, module):
        """Validation step completes without error."""
        batch = self._make_batch()
        module.validation_step(batch, 0)

    def test_loss_is_finite(self, module):
        """Loss doesn't produce NaN or Inf."""
        batch = self._make_batch()
        loss = module.training_step(batch, 0)
        assert torch.isfinite(loss)
