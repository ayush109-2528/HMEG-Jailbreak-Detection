from scripts.compare_public_candidate import deployment_gate


def test_lower_fpr_cannot_hide_recall_regression():
    old={'fpr':.2,'recall':.9,'f1':.8}
    candidate={'fpr':.1,'recall':.89,'f1':.85}
    checks=deployment_gate(old,candidate,{'fpr':.3},{'fpr':.2})
    assert not all(checks.values()) and not checks['recall_not_lower']


def test_equal_fpr_does_not_count_as_improvement():
    old={'fpr':.2,'recall':.9,'f1':.8}
    assert not all(deployment_gate(old,old,{'fpr':.3},{'fpr':.3}).values())


def test_ood_regression_blocks_candidate():
    old={'fpr':.2,'recall':.9,'f1':.8}
    better={'fpr':.1,'recall':.91,'f1':.85}
    assert not all(deployment_gate(old,better,{'fpr':.3},{'fpr':.4}).values())
    assert all(deployment_gate(old,better,{'fpr':.3},{'fpr':.2}).values())


from scripts.compare_public_candidate import english_ood_gate


def test_english_external_gate_rejects_missing_or_regressed_metrics():
    old={'fpr':.2,'recall':.9,'f1':.8}
    assert not all(english_ood_gate(old,{'fpr':.1,'recall':.89,'f1':.85}).values())
    assert not all(english_ood_gate(old,{'fpr':None,'recall':.95,'f1':.85}).values())
    assert not all(english_ood_gate(old,{'fpr':float('nan'),'recall':.95,'f1':.85}).values())
    assert all(english_ood_gate(old,{'fpr':.1,'recall':.95,'f1':.85}).values())
