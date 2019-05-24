#coding=utf-8
import warnings
import os
import fire
import paramiko
import subprocess
import re
from pandas import DataFrame

warnings.filterwarnings("ignore")

filename = 'features.csv'


def decorate_for_function_name(func):
    def func_wrapper(self):
        print("================%s============" % func.__name__)
        return func(self)
    return func_wrapper

class GetFeatures(object):

    def run_cmd(self, cmd):
        print("command: " + cmd)

        if not self.client or self.client == 'localhost':
            # print("run on localhost")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output = stdout.decode('ascii').strip().strip("\n")
            print("output: " + output + "\n")
            if not output:
                print("stderr:" + stderr.decode('ascii').strip().strip("\n") + "\n")
        else:
            # print("run from remote server")
            try:
                client = paramiko.SSHClient()
                client.load_system_host_keys()
                client.set_missing_host_key_policy(paramiko.WarningPolicy)

                client.connect(self.client, port=22, username=self.user, password=self.pwd)

                stdin, stdout, stderr = client.exec_command(cmd)
                output = stdout.read().decode('ascii').strip().strip("\n")
                print("output: " + output + "\n")
                if not output:
                    print("stderr:" + stderr.decode('ascii').strip().strip("\n") + "\n")
            finally:
                client.close()
        return output

    @decorate_for_function_name
    def get_volume_size(self):
        self.volume_size = self.run_cmd("df -Th %s | grep %s | awk -F' ' '{print $3}'" % (self.volume, self.volume.strip('/')))

    @decorate_for_function_name
    def get_volume_used_size(self):
        self.volume_used_size = self.run_cmd("df -Th %s | grep %s | awk -F' ' '{print $4}'" % (self.volume, self.volume.strip('/')))

    @decorate_for_function_name
    def get_fs_type(self):
        self.fs_type = self.run_cmd("df -Th %s | grep %s | awk -F' ' '{print $2}'" % (self.volume, self.volume.strip('/')))

    @decorate_for_function_name
    def get_file_count(self):
        self.file_count = self.run_cmd("find %s -type f | wc -l" % (self.volume))

    @decorate_for_function_name
    def get_file_size(self):
        pass

    @decorate_for_function_name
    def get_total_memory(self):
        self.total_memory = self.run_cmd("free -h | grep Mem | awk -F' ' '{print $2}'")

    @decorate_for_function_name
    def get_free_memory(self):
        self.free_memory = self.run_cmd("free -h | grep Mem | awk -F' ' '{print $4}'")

    @decorate_for_function_name
    def get_shared_memory(self):
        self.shared_memory = self.run_cmd("free -h | grep Mem | awk -F' ' '{print $5}'")

    @decorate_for_function_name
    def get_cached_memory(self):
        self.cached_memory = self.run_cmd("free -h | grep Mem | awk -F' ' '{print $6}'")

    @decorate_for_function_name
    def get_cpu_count(self):
        self.cpu_count = self.run_cmd("lscpu | egrep '^CPU\(s\)' | awk -F' ' '{print $NF}'")

    @decorate_for_function_name
    def get_average_cpu_usage_on_previous_one_minute(self):
        self.average_cpu_usage_on_previous_one_minute = self.run_cmd("uptime | awk -F'load average:' '{print $NF}' | awk -F',' '{print $1}'")

    @decorate_for_function_name
    def get_average_cpu_usage_on_previous_five_minute(self):
        self.average_cpu_usage_on_previous_five_minute = self.run_cmd("uptime | awk -F'load average:' '{print $NF}' | awk -F',' '{print $2}'")

    @decorate_for_function_name
    def get_average_cpu_usage_on_previous_fifteen_minute(self):
        self.average_cpu_usage_on_previous_fifteen_minute = self.run_cmd("uptime | awk -F'load average:' '{print $NF}' | awk -F',' '{print $3}'")

    @decorate_for_function_name
    def get_extents_and_backup_time(self):
        backup_stdout = self.run_cmd(self.backup_cmd)
        extents = re.search("predict metrics: \[(.*)\]", backup_stdout)
        if not extents:
            print("Not able to get extents, backup might fail, please check and re-rerun")
            exit(1)
        print(extents[0])
        extents = re.search("\[(.*)\]", extents[0])
        print(extents[0])
        # NE: number extents
        # NR: number bytes read
        # NW: number bytes written
        # SSHM: square sum of head movement in bytes
        # SSR: square sum of read in bytes
        # ET: backup elapsed time in USecs
        self.num_extents, self.num_bytes_read, self.num_bytes_write, self.square_sum_head_move, self.square_sum_read, \
        self.backup_elapsed_time = extents[0].strip('[').strip(']').split(':')

    @decorate_for_function_name
    def get_volume_list(self):
        self.volume_list = self.run_cmd("ls %s" % self.volume_parent_dir)

    @decorate_for_function_name
    def export_data(self):
        features = {'volume_size': [self.volume_size],
                    'volume_used_size': self.volume_used_size,
                    'fs_type': self.fs_type,
                    'file_count': self.file_count,
                    'total_memory': self.total_memory,
                    'free_memory': self.free_memory,
                    'shared_memory': self.shared_memory,
                    'cached_memory': self.cached_memory,
                    'cpu_count': self.cpu_count,
                    'average_cpu_usage_on_previous_one_minute': self.average_cpu_usage_on_previous_one_minute,
                    'average_cpu_usage_on_previous_five_minute': self.average_cpu_usage_on_previous_five_minute,
                    'average_cpu_usage_on_previous_fifteen_minute': self.average_cpu_usage_on_previous_fifteen_minute,
                    'num_extents': self.num_extents,
                    'num_bytes_read': self.num_bytes_read,
                    'num_bytes_write': self.num_bytes_write,
                    'square_sum_head_move': self.square_sum_head_move,
                    'square_sum_read': self.square_sum_read,
                    'backup_elapsed_time': self.backup_elapsed_time
                    }
        df = DataFrame(features, index=[0])
        df.to_csv(filename, mode='a', index=False,  header=(not os.path.exists(filename)))
        print("Please see data in file: %s" % filename)

    def single_volume(self, volume, backup_cmd, client='', user='', pwd=''):
        self.volume = volume
        self.client = client
        self.user = user
        self.pwd = pwd
        self.backup_cmd = backup_cmd

        if not self.client or self.client == 'localhost':
            print("client: localhost")
        else:
            print("client: " + self.client)
        print("username: " + self.user)
        print("password: " + self.pwd)
        print("volume: " + self.volume)
        print("backup_cmd: " + self.backup_cmd)
        print("")
        self.get_volume_size()
        self.get_volume_used_size()
        self.get_fs_type()
        self.get_file_count()
        self.get_total_memory()
        self.get_free_memory()
        self.get_shared_memory()
        self.get_cached_memory()
        self.get_cpu_count()
        self.get_average_cpu_usage_on_previous_one_minute()
        self.get_average_cpu_usage_on_previous_five_minute()
        self.get_average_cpu_usage_on_previous_fifteen_minute()
        self.get_extents_and_backup_time()
        self.export_data()

    def multi_volume(self, volume_parent_dir, backup_cmd, client='', user='', pwd=''):
        self.volume_parent_dir = volume_parent_dir
        self.client = client
        self.user = user
        self.pwd = pwd
        self.backup_cmd = backup_cmd

        print("client: " + self.client)
        print("username: " + self.user)
        print("password: " + self.pwd)
        print("volume_parent_dir: " + self.volume_parent_dir)
        print("backup_cmd: " + self.backup_cmd)
        print("")
        self.get_volume_list()

        for volume in self.volume_list.split():
            volume = self.volume_parent_dir.rstrip('/') + '/' + volume
            self.single_volume(volume, self.backup_cmd + ' ' + volume, self.client, self.user, self.pwd)

if __name__ == "__main__":
    get_features = GetFeatures()
    fire.Fire(get_features)
