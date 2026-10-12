from mmengine.logging import MMLogger

from cytospm.evaluation.organ_restricted_metric import (
    OrganRestrictedCocoMetric,
    _zero_metric_results,
)


HEADLINE_KEYS = {
    'overall/macro_mAP',
    'overall/macro_mAP_50',
    'overall/macro_mAP_75',
    'overall/all_class_mAP',
    'overall/instance_weighted_mAP',
}


def test_zero_metric_schema_includes_all_headline_metrics():
    result = _zero_metric_results()

    assert HEADLINE_KEYS <= result.keys()
    assert all(value == 0.0 for value in result.values())


def test_empty_results_return_stable_zero_schema():
    MMLogger.get_instance('organ-metric-test', log_file=None)
    metric = object.__new__(OrganRestrictedCocoMetric)
    metric._assert_class_alignment = lambda: None

    result = metric.compute_metrics([])

    assert HEADLINE_KEYS <= result.keys()
    assert result['overall/macro_mAP'] == 0.0
    assert result['overall/macro_mAP_50'] == 0.0
    assert result['overall/macro_mAP_75'] == 0.0


def test_samples_with_no_detections_return_same_zero_schema(tmp_path):
    empty_predictions = tmp_path / 'empty.bbox.json'
    empty_predictions.write_text('[]', encoding='utf-8')

    metric = object.__new__(OrganRestrictedCocoMetric)
    metric._assert_class_alignment = lambda: None
    metric.outfile_prefix = str(tmp_path / 'results')
    metric._coco_api = object()
    metric.cat_ids = []
    metric.img_ids = []
    metric.dataset_meta = {'classes': []}
    metric._exclude_class_names = []
    metric.results2json = lambda predictions, prefix: {
        'bbox': str(empty_predictions)
    }

    result = metric.compute_metrics([({}, {'pred_instances': {}})])

    assert result == _zero_metric_results()
