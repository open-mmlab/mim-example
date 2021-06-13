import mmcv
import os.path as osp
from mmcv.utils import print_log
from mmseg.datasets.builder import DATASETS
from mmseg.datasets import CustomDataset
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class NuImagesDataset(CustomDataset):
    CLASSES = ()

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        # Here we take ann_dir as the annotation path
        annotations = mmcv.load(ann_dir)
        ann_root = osp.split(ann_dir)[0]
        img_infos = []
        for img in annotations['images']:
            img_info = dict(filename=img['file_name'])
            seg_map = img_info['filename'].replace(img_suffix, seg_map_suffix)
            img_info['ann'] = dict(
                seg_map=osp.join(ann_root, 'semantic_masks', seg_map))
            img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {ann_dir}',
            logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        super(NuImagesDataset, self).pre_pipeline(results)
        results['seg_prefix'] = ''
