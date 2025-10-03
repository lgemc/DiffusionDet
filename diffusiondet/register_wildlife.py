"""
Register WildlifeMapper dataset in COCO format for DiffusionDet.
"""
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_wildlife_dataset():
    """
    Register WildlifeMapper dataset in COCO format.
    Supports both 640x640 and 1024x1024 patch sizes.
    """
    # Define dataset paths - adjust these to match your data location
    # Using 640x640 patches to match DiffusionDet config
    annotation_root = "/home/lmanrique/Do/WildlifeMapper/data/processed/640_640"
    image_root = "/home/lmanrique/Do/WildlifeMapper/data/intermediate/patches_640_640"

    # Wildlife species classes (update with actual species names if available)
    wildlife_classes = [
        "species_1", "species_2", "species_3",
        "species_4", "species_5", "species_6"
    ]

    # Register train, validation and test datasets
    for split in ["train", "val", "test"]:
        dataset_name = f"wildlifemapper_{split}"
        json_file = os.path.join(annotation_root, f"{split}_coco.json")
        image_dir = os.path.join(image_root, split)

        register_coco_instances(
            dataset_name,
            {},
            json_file,
            image_dir,
        )

        # Set metadata
        MetadataCatalog.get(dataset_name).set(
            thing_classes=wildlife_classes,
            evaluator_type="coco",
        )


# Register the datasets when this module is imported
register_wildlife_dataset()
