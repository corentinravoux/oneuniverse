"""Phase 18 T5 — ClassificationPdfSpec sub-spec."""
import pytest

from oneuniverse.data.classification_pdf import ClassificationPdfSpec


def test_defaults_and_classes_required():
    spec = ClassificationPdfSpec(classes=("galaxy", "qso", "star"))
    assert spec.value_column == "class_pdf_values"
    assert spec.parameterisation == "categorical"
    assert spec.n_classes == 3


def test_rejects_empty_classes():
    with pytest.raises(ValueError, match="classes"):
        ClassificationPdfSpec(classes=())


def test_rejects_unknown_parameterisation():
    with pytest.raises(ValueError, match="parameterisation"):
        ClassificationPdfSpec(
            classes=("a", "b"), parameterisation="mystery",
        )


def test_roundtrip():
    spec = ClassificationPdfSpec(
        classes=("galaxy", "qso", "star", "agn"),
        parameterisation="categorical",
        value_column="p_class",
    )
    d = spec.to_dict()
    restored = ClassificationPdfSpec.from_dict(d)
    assert restored == spec
