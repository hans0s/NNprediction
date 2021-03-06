# encoding: utf-8

import logger_utils
import fire
import volume_utils
import config

logger = logger_utils.get_logger("backup_excutor")


def create_volumes(volumes, fs="ext3"):
    for mount_point, size in volumes.items():
        volume_utils.create_volume(volume_size=size, mount_point=mount_point, fs_type=fs)
    return True


def fill(targets=config.VOLUMES_FILL_TARGET):
    for volume, target in targets.items():
        logger.info("Start Fill %s to %s..." % (volume, target))
        volume_utils.fill_volume(volume=volume, percentage=target)
        logger.info("Completed Fill %s to %s." % (volume, target))
    volume_utils.execute("df -h")


def create():
    do = raw_input('You are creating new volumes via your configuration. Are you sure this operation:(yes/[no]) ')
    if do == "yes":
        create_volumes(volumes=config.EXT3_VOLUMES, fs="ext3")
        create_volumes(volumes=config.EXT4_VOLUMES, fs="ext4")
        logger.info("Volumes creation completed.")
        lvs = volume_utils.list_lv()
        return lvs


def change(targets=config.CHANGE_FILES_TARGET):
    for volume, target_kwargs in targets.items():
        volume_info = volume_utils.get_volume_status(volume)
        if volume_info["used_percent"] >= 99:
            logger.warn("The volume %s usage is more than 99 percent. Only perform the delete operation..." % volume)
            target_kwargs["c"] = 0
            target_kwargs["u"] = 0
        if volume_info["used_percent"] <= 1:
            logger.warn("The volume %s usage is less than 1 percent. Do not perform the delete operation..." % volume)
            target_kwargs["d"] = 0
        volume_utils.change_files(file_dir=volume, **target_kwargs)


if __name__ == "__main__":
    fire.Fire()
