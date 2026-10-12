"""Read the organ of an image from its path.

The organ (specimen type) is known in clinical practice, so it is available at
inference. It is taken from a directory name in the image path, e.g.
``images/respiratory_tract/xxx.jpg``. The transform adds ``organ_id`` and
``organ_name``; add both to ``PackDetInputs.meta_keys``.
"""
import json
from pathlib import Path

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class OrganExtractor:
    """Extract clinical organ id from the image path.

    Args:
        taxonomy_path: taxonomy JSON providing ``organ_to_id``.
        organ_segment_map: optional explicit mapping {path_token: organ_name}.
            By default each organ_name is matched as its space-to-underscore
            form (e.g. 'respiratory tract' → 'respiratory_tract').
        strict: if True, raise on unmatched paths. If False, write organ_id=-1.
    """

    def __init__(self,
                 taxonomy_path: str,
                 organ_segment_map: dict = None,
                 strict: bool = True):
        tax = json.loads(Path(taxonomy_path).read_text())
        self.organ_to_id = dict(tax['organ_to_id'])
        self.strict = strict

        if organ_segment_map is None:
            organ_segment_map = {
                organ.replace(' ', '_'): organ
                for organ in self.organ_to_id
            }
        # Path tokens must not contain each other, or matching is ambiguous.
        tokens = sorted(organ_segment_map.keys(), key=len)
        for i, t in enumerate(tokens):
            for u in tokens[i + 1:]:
                if t in u or u in t:
                    raise ValueError(
                        f'OrganExtractor: organ tokens {t!r} and {u!r} are '
                        f'prefix/substring of each other — path matching is '
                        f'ambiguous. Provide explicit organ_segment_map.')
        self.segment_to_organ = organ_segment_map  # path token → canonical name

    def __call__(self, results: dict) -> dict:
        img_path = results.get('img_path') or results.get('filename')
        if img_path is None:
            if self.strict:
                raise ValueError('OrganExtractor: no img_path in results')
            results['organ_id'] = -1
            results['organ_name'] = ''
            return results

        organ_name = None
        for seg in str(img_path).split('/'):
            if seg in self.segment_to_organ:
                organ_name = self.segment_to_organ[seg]
                break

        if organ_name is None:
            if self.strict:
                raise ValueError(
                    f'OrganExtractor: no organ segment in img_path: {img_path}\n'
                    f'  expected one of: {sorted(self.segment_to_organ.keys())}')
            results['organ_id'] = -1
            results['organ_name'] = ''
            return results

        results['organ_id'] = self.organ_to_id[organ_name]
        results['organ_name'] = organ_name
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"organs={list(self.organ_to_id.keys())}, strict={self.strict})")
