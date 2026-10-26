from check_docs import find_forbidden, find_private_paths, find_unverifiable_bib_entries, main


def test_flags_results_numbers():
    for bad in ["R^2 = 0.93", "R² ≈ 0.69", ":math:`R^2 = 0.978`", "coverage of 0.63",
                "RMSE of 1.2", "6893 img/s", "0.145 ms per image",
                r":math:`R^2 \approx 0.69`", r"R^2 \sim 0.8"]:
        assert find_forbidden(bad), bad


def test_allows_definitions_and_ranges():
    for ok in [r":math:`R^2 = 1 - \mathrm{SS}_\mathrm{res}/\mathrm{SS}_\mathrm{tot}`",
               r"k_\mathrm{min} \in [1, 32]", "a 68% nominal interval", "5 members", "128×128"]:
        assert not find_forbidden(ok), ok


def test_flags_nonexistent_variance_head():
    assert find_forbidden("the model's aleatoric head predicts a variance")


def test_flags_private_paths():
    assert find_private_paths("see archive/docs/RESEARCH_JOURNEY.md")
    assert find_private_paths("outputs/comparison/local_v2")
    assert find_private_paths("see `src/dataset_generator.py`")
    assert find_private_paths("(`notebooks/turbulens_multitask_results.ipynb`)")
    assert not find_private_paths("the HF repo Davas1410/turbulens-ensembles")


def test_flags_bib_entries_without_identifier():
    bib = "@article{good, doi={10.1/x}}\n@article{bad, title={No id}}"
    assert find_unverifiable_bib_entries(bib) == ["bad"]


def test_main_reports_hits_and_honours_exclude(tmp_path, capsys):
    (tmp_path / "ok.rst").write_text("Clean page.\n", encoding="utf-8")
    (tmp_path / "bad.rst").write_text("Title\n=====\n\nWe reach R^2 = 0.93.\n", encoding="utf-8")
    (tmp_path / "references.bib").write_text("", encoding="utf-8")

    assert main(["--root", str(tmp_path)]) == 1
    assert "bad.rst:4" in capsys.readouterr().out
    assert main(["--root", str(tmp_path), "--exclude", "bad.rst"]) == 0
    assert "check_docs: OK" in capsys.readouterr().out


def test_main_scans_package_docstrings(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "page.rst").write_text("Clean.\n", encoding="utf-8")
    package = tmp_path / "pkg"
    (package / "tests").mkdir(parents=True)
    (package / "mod.py").write_text('"""Accuracy note: k_max R^2 dropped from 0.91 to 0.27."""\n', encoding="utf-8")
    (package / "tests" / "test_mod.py").write_text('"""see archive/old.md"""\n', encoding="utf-8")

    assert main(["--root", str(docs), "--package", str(package)]) == 1
    out = capsys.readouterr().out
    assert "mod.py:1" in out
    assert "test_mod.py" not in out


def test_flags_common_results_phrasings():
    for bad in ["On GASS data R^2 = 0.93", "the LOSS reached R^2 = 0.9", "RMSE: 1.2", "R^2 was 0.93",
                "k_max R^2 0.91", "R2 = 0.93", "MAE 0.4", "coverage of 63%", "covers 68% of truths",
                "7000 images per second"]:
        assert find_forbidden(bad), bad


def test_definition_exemption_needs_latex_markers():
    assert not find_forbidden(r"R^2 = 1 - \mathrm{SS}_\mathrm{res}/\mathrm{SS}_\mathrm{tot}")
    assert find_forbidden("GASS fields give R^2 = 0.8")
