import os.path as osp

import mmcv
from mmcv.utils import print_log
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class NuImagesDataset(CustomDataset):
    CLASSES = (
        'animal',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.stroller',
        'human.pedestrian.wheelchair',
        'movable_object.barrier',
        'movable_object.debris',
        'movable_object.pushable_pullable',
        'movable_object.trafficcone',
        'static_object.bicycle_rack',
        'vehicle.bicycle',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.car',
        'vehicle.construction',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.motorcycle',
        'vehicle.trailer',
        'vehicle.truck',
        'flat.driveable_surface',
        'vehicle.ego',
    )

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
        annotations = mmcv.load(split)
        img_infos = []
        for img in annotations['images']:
            img_info = dict(filename=img['file_name'])
            seg_map = img_info['filename'].replace(img_suffix, seg_map_suffix)
            img_info['ann'] = dict(seg_map=osp.join('semantic_masks', seg_map))
            img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {ann_dir}',
            logger=get_root_logger())
        return img_infos
