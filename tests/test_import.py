def test_import():
    import ww_pgd
    assert hasattr(ww_pgd, "WWPGDWrapper")
    assert hasattr(ww_pgd, "WWTailConfig")
