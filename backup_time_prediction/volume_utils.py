#!/usr/bin/env python
# encoding: utf-8

import subprocess
import datetime
import os
import logger_utils
import fire
from random import randint, sample
from lvm2py import LVM

logger = logger_utils.get_logger("volume_utils")


def create_volume(volume_size, mount_point=None, fs_type="ext3", vg="fsa", volume_name=None):
    if volume_name is None:
        volume_name = "%s-%s" % (volume_size, fs_type)
    if mount_point is None:
        mount_point = "/lvm/%s-%s" % (vg, volume_name)
    if is_mount_point_available(mount_point):
        logger.info("Mount point %s is exist. Please remove it first." % mount_point)
        return mount_point
    volume_path = get_lv_path(lv_name=volume_name, vg_name=vg)
    if not volume_path:
        volume_path = create_lv(volume_name=volume_name, size=volume_size, vg_name=vg)
        if volume_path:
            formatted = format_lv(volume_path=volume_path, fs_type=fs_type)
        else:
            logger.error("Volume %s creation failed." % volume_name)
            return None
        if not formatted:
            logger.error("Volume %s format failed." % volume_path)
            return None

    mounted = mount_lv(volume_path=volume_path, mount_point=mount_point)
    if not mounted:
        logger.error("Volume %s mount to %s failed." % (volume_path, mount_point))
        return None
    else:
        logger.info("Volume %s created" % mount_point)
        return mount_point


def is_mount_point_available(mount_point):
    cmd = "df -h | grep %s" % (mount_point)
    status, output = execute(cmd, ignore_error=True)
    if status == 0:
        return True
    else:
        return False


def get_volume_status(mount_point):
    cmd = "df | grep %s" % (mount_point)
    status, output = execute(cmd, ignore_error=True)
    if status == 0:
        try:
            status_list = output.split()[1:5]
            detail = dict(zip(["1k_blocks", "used", "available", "used_percent"],
                              map(lambda x: int(x.replace("%", "")), status_list)))  # convert to integer
        except Exception as e:
            logger.error("Get the volume info of %s failed. ERROR: %s" % (mount_point, e.message))
            return None
        logger.info("%s details: %s" % (mount_point, detail))
        return detail
    else:
        logger.error("Volume mount point %s is not available." % mount_point)
        return None


def list_lv(vg_name="fsa"):
    lvm = LVM()
    vg = lvm.get_vg(name=vg_name)
    lv_names = []
    for lv in vg.lvscan():
        lv_names.append(lv.name)
    return lv_names


def get_lv_path(lv_name, vg_name="fsa"):
    logger.info("Getting the logical volume path for %s..." % lv_name)
    lvm = LVM()
    vg = lvm.get_vg(name=vg_name)
    try:
        vg.get_lv(name=lv_name)
    except Exception as e:
        logger.warn("Logic volume %s is not exist" % lv_name)
        return None

    return "/dev/%s/%s" % (vg_name, lv_name)


def execute(cmd, ignore_error=False):
    print("INFO: Execute command  '%s'" % cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = "".join(process.stdout.readlines())
    process.communicate()
    status = process.returncode
    if status != 0:
        log = logger.warning if ignore_error else logger.error
        log("Run command '%s' failed." % cmd)
        log("ERROR code: %s. OUTPUT: %s" % (status, output))
    else:
        logger.info("Run command '%s' successful." % cmd)
        logger.info("%s" % output)
    return status, output


def create_lv(volume_name, size, vg_name="fsa"):
    cmd = "lvcreate -L %s -n %s %s" % (size, volume_name, vg_name)
    status, output = execute(cmd)
    volume_path = None if status else "/dev/%s/%s" % (vg_name, volume_name)
    if volume_path:
        logger.info("LV %s created successfully" % volume_name)
    return volume_path


def remove_lv(volume_name, vg_name="fsa"):
    cmd = "lvremove /dev/%s/%s -f" % (vg_name, volume_name)
    status, output = execute(cmd)
    return True if status == 0 else False


def format_lv(volume_path, fs_type="ext3"):
    cmd = "mkfs.%s %s" % (fs_type, volume_path)
    status, output = execute(cmd)
    return True if status == 0 else False


def mount_lv(volume_path, mount_point):
    if not os.path.exists(mount_point):
        os.makedirs(mount_point)
    cmd = "mount %s %s" % (volume_path, mount_point)
    status, output = execute(cmd)
    return True if status == 0 else False


def generate_random_content_file(file_path, block_count=1000, block_size_k=1, seek=0):
    block_count = int(block_count)
    block_size_k = int(block_size_k)
    file_dir = os.path.dirname(file_path)
    if not file_dir:
        file_dir = os.path.curdir
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if os.path.isdir(file_path):
        logger.error("Path %s is a folder." % file_path)
        return None
    cmd = "dd bs=%dk count=%d if=/dev/urandom of=%s seek=%s> /dev/null 2>&1" % (
        block_size_k, block_count, file_path, seek)

    execute("ls -lh %s" % file_path)
    status, output = execute(cmd)
    if status:
        logger.error("File %s is not generated" % file_path)
        return None
    execute("ls -lh %s" % file_path)
    return file_path


def get_file_size_k(file_path):
    fsize = os.path.getsize(file_path)
    return fsize / 1024


def change_files(file_dir, count=1, cr=0.01, bsk=100):
    """

    :param file_dir: Which director you wang to change
    :param count: How many files you want to change
    :param cr: Change Rate of the files.
    :param bsk: Block Size, Unit is KB
    :return: The real changed file count
    """
    count = int(count)
    cr = float(cr)
    bsk = int(bsk)

    logger.info("Change files in %s..." % file_dir)
    files = os.listdir(file_dir)
    files_count = len(files)
    changed_count = min(files_count, count)
    file_samples = sample(population=files, k=changed_count)
    actually_changed_count = 0
    for file_sample in file_samples:
        file_sample = os.path.join(file_dir, file_sample)
        if os.path.isdir(file_sample):
            logger.warn("%s is a folder. skip it." % file_sample)
            continue
        change_file_by_change_rate(file_path=file_sample, change_rate=cr, block_size_k=bsk)
        actually_changed_count += 1
    logger.info("Change %s files in %s completed." % (actually_changed_count, file_dir))
    return actually_changed_count


def change_file_by_blocks_count(file_path, block_count=10, block_size_k=1):
    file_size_k = get_file_size_k(file_path)
    file_current_blocks = file_size_k / block_size_k
    return generate_random_content_file(file_path=file_path, block_count=block_count, block_size_k=block_size_k,
                                        seek=file_current_blocks)


def change_file_by_change_rate(file_path, change_rate=0.01, block_size_k=1):
    logger.info("Change file %s..." % file_path)
    file_size_k = get_file_size_k(file_path)
    file_current_blocks = file_size_k / block_size_k
    changed_blocks = int(file_current_blocks * change_rate)
    return generate_random_content_file(file_path=file_path, block_count=changed_blocks, block_size_k=block_size_k,
                                        seek=file_current_blocks)


def fill_volume(volume, percentage=10, max_single_file_size=1000 * 100, min_single_file_size=10, accuracy_kb=10000, ):
    volume_detail = get_volume_status(volume)
    target_size = (volume_detail["used"] + volume_detail["available"]) * percentage / 100

    while volume_detail["used_percent"] < percentage:
        diff_size = target_size - volume_detail["used"]
        logger.debug(
            "Target is %s percent, Target size is %s, used size is %s " % (
                percentage, target_size, volume_detail["used"]))
        logger.debug("We need fill %sKB to achieve the target" % diff_size)
        max_file_size = min(diff_size, max_single_file_size)  # file count control.
        if min_single_file_size >= max_file_size - accuracy_kb:
            logger.info("The file size accuracy is OK. Current volume status is %s" % volume_detail)
            break
        file_size = randint(min_single_file_size, max_file_size)
        block_size = accuracy_kb / 10
        block_count = file_size / block_size
        filename = "bc%s-bs%s-%s" % (block_count, block_size, datetime.datetime.now().isoformat())
        filename = os.path.join(volume, filename)
        generate_random_content_file(file_path=filename, block_count=block_count, block_size_k=block_size)
        volume_detail = get_volume_status(volume)
    logger.info("Fill volume completed.")


def set_read_buffer():
    pass


# if __name__ == "__main__":
#     fire.Fire()
