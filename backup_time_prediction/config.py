# encoding: utf-8

EXT3_VOLUMES = {
    "/lvm/fsa-1G-ext3": "1G",
    "/lvm/fsa-2G-ext3": "2G",
    "/lvm/fsa-5G-ext3": "5G",
    "/lvm/fsa-10G-ext3": "10G",
    "/lvm/fsa-20G-ext3": "20G",
    "/lvm/fsa-50G-ext3": "50G",
    "/lvm/fsa-100G-ext3": "100G",
}

EXT4_VOLUMES = {
    "/lvm/fsa-1G-ext4": "1G",
    "/lvm/fsa-2G-ext4": "2G",
    "/lvm/fsa-5G-ext4": "5G",
    "/lvm/fsa-10G-ext4": "10G",
    "/lvm/fsa-20G-ext4": "20G",
    "/lvm/fsa-50G-ext4": "50G",
    "/lvm/fsa-100G-ext4": "100G",
}

VOLUMES_FILL_TARGET = {  # Specify the target percentage need to fill of the volume size
    "/lvm/fsa-1G-ext3": 65,
    "/lvm/fsa-2G-ext3": 50,
    "/lvm/fsa-5G-ext3": 50,
    "/lvm/fsa-10G-ext3": 50,
    "/lvm/fsa-20G-ext3": 50,
    "/lvm/fsa-50G-ext3": 10,
    "/lvm/fsa-100G-ext3": 10,
    "/lvm/fsa-1G-ext4": 50,
    "/lvm/fsa-2G-ext4": 50,
    "/lvm/fsa-5G-ext4": 50,
    "/lvm/fsa-10G-ext4": 50,
    "/lvm/fsa-20G-ext4": 50,
    "/lvm/fsa-50G-ext4": 20,
    "/lvm/fsa-100G-ext4": 5,

}

CHANGE_FILES_TARGET = {  # Specify changed files count, change rate for each file, and block size
    "/lvm/fsa-1G-ext3": {"u": 10, "c": 15, "d": 10, "cr": 0.05, "bsk": 1},
    "/lvm/fsa-2G-ext3": {"u": 10, "c": 15, "d": 10, "cr": 0.05, "bsk": 1},
    "/lvm/fsa-5G-ext3": {"u": 10, "c": 15, "d": 10, "cr": 0.05, "bsk": 1},
    "/lvm/fsa-10G-ext3": {"u": 20, "c": 30, "d": 10, "cr": 0.01, "bsk": 1},
    "/lvm/fsa-20G-ext3": {"u": 20, "c": 30, "d": 10, "cr": 0.01, "bsk": 1},
    "/lvm/fsa-50G-ext3": {"u": 40, "c": 50, "d": 20, "cr": 0.005, "bsk": 1},
    "/lvm/fsa-100G-ext3": {"u": 40, "c": 50, "d": 20, "cr": 0.003, "bsk": 1},

    "/lvm/fsa-1G-ext4": {"u": 10, "c": 15, "d": 10, "cr": 0.05, "bsk": 1},
    "/lvm/fsa-2G-ext4": {"u": 10, "c": 15, "d": 10, "cr": 0.04, "bsk": 1},
    "/lvm/fsa-5G-ext4": {"u": 15, "c": 30, "d": 10, "cr": 0.03, "bsk": 1},
    "/lvm/fsa-10G-ext4": {"u": 20, "c": 50, "d": 10, "cr": 0.02, "bsk": 1},
    "/lvm/fsa-20G-ext4": {"u": 20, "c": 50, "d": 10, "cr": 0.01, "bsk": 1},
    "/lvm/fsa-50G-ext4": {"u": 30, "c": 100, "d": 10, "cr": 0.005, "bsk": 1},
    "/lvm/fsa-100G-ext4": {"u": 30, "c": 100, "d": 10, "cr": 0.003, "bsk": 1},
}
