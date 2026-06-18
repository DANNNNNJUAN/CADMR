from cadmr.scope import canonicalize_scope, canonicalize_scopes


def test_canonicalize_scope_normalizes_format_without_semantic_aliases():
    assert canonicalize_scope("dinner") == ["dinner"]
    assert canonicalize_scope("饮食选择") == ["饮食选择"]
    assert canonicalize_scope("preferences") == ["preferences"]
    assert canonicalize_scope("Diet Preference") == ["diet_preference"]


def test_canonicalize_scopes_deduplicates_preserving_order():
    scopes = canonicalize_scopes(["dinner", "diet", "饮食偏好", "preferences"])

    assert scopes == ["dinner", "diet", "饮食偏好", "preferences"]


def test_canonicalize_scope_does_not_infer_compound_semantics():
    assert canonicalize_scope("长期记忆系统demo") == ["长期记忆系统demo"]
    assert canonicalize_scope("真实用户数据") == ["真实用户数据"]
