import os
from typing import List, Union

#import numba


from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox

from aydin.util.log.log import lprint
from aydin.util.misc.units import human_readable_byte_size


class SystemSummaryWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)

        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)

        # CPU summary
        self.cpu_group_box = QGroupBox("CPU Summary")
        self.cpu_group_box_layout = QVBoxLayout()
        self.cpu_group_box_layout.setSpacing(0)
        self.cpu_group_box_layout.setAlignment(Qt.AlignTop)
        self.cpu_group_box.setLayout(self.cpu_group_box_layout)

        # CPU freq
        self.cpu_freq_stats_label = QLabel(
            f"Current CPU frequency:\t {self._cpu_freq()}", self
        )
        self.cpu_group_box_layout.addWidget(self.cpu_freq_stats_label)

        # Number of cores
        self.nb_cores_label = QLabel(
            f"Number of CPU cores:\t {self._number_of_cores()}", self
        )
        self.cpu_group_box_layout.addWidget(self.nb_cores_label)
        if (self._number_of_cores_int()) < 4:
            self.nb_cores_label.setStyleSheet("QLabel {color: red;}")
        elif (self._number_of_cores_int()) <= 6:
            self.nb_cores_label.setStyleSheet("QLabel {color: orange;}")
        else:
            self.nb_cores_label.setStyleSheet("QLabel {color: green;}")

        self.cpu_load_values = self._cpu_load_values()

        self.cpu_load_label0 = QLabel(
            f"CPU load over last 1min:\t {'100.0+' if self.cpu_load_values[0] >= 100.0 else round(self.cpu_load_values[0], 2)}%",
            self,
        )
        self.cpu_group_box_layout.addWidget(self.cpu_load_label0)

        self.cpu_load_label1 = QLabel(
            f"CPU load over last 5mins:\t {'100.0+' if self.cpu_load_values[1] >= 100.0 else round(self.cpu_load_values[1], 2)}%",
            self,
        )
        self.cpu_group_box_layout.addWidget(self.cpu_load_label1)

        self.cpu_load_label2 = QLabel(
            f"CPU load over last 15mins:\t {'100.0+' if self.cpu_load_values[2] >= 100.0 else round(self.cpu_load_values[2], 2)}%",
            self,
        )
        self.cpu_group_box_layout.addWidget(self.cpu_load_label2)

        if self.cpu_load_values[0] >= 30:
            self.cpu_load_label0.setStyleSheet("QLabel {color: red;}")
        elif self.cpu_load_values[0] > 15:
            self.cpu_load_label0.setStyleSheet("QLabel {color: orange;}")
        else:
            self.cpu_load_label0.setStyleSheet("QLabel {color: green;}")
        if self.cpu_load_values[1] >= 30:
            self.cpu_load_label1.setStyleSheet("QLabel {color: red;}")
        elif self.cpu_load_values[1] > 15:
            self.cpu_load_label1.setStyleSheet("QLabel {color: orange;}")
        else:
            self.cpu_load_label1.setStyleSheet("QLabel {color: green;}")
        if self.cpu_load_values[2] >= 30:
            self.cpu_load_label2.setStyleSheet("QLabel {color: red;}")
        elif self.cpu_load_values[2] > 15:
            self.cpu_load_label2.setStyleSheet("QLabel {color: orange;}")
        else:
            self.cpu_load_label2.setStyleSheet("QLabel {color: green;}")

        # Memory summary
        self.memory_group_box = QGroupBox("Memory Summary")
        self.memory_group_box_layout = QVBoxLayout()
        self.memory_group_box_layout.setSpacing(0)
        self.memory_group_box_layout.setAlignment(Qt.AlignTop)
        self.memory_group_box.setLayout(self.memory_group_box_layout)

        self.free_memory_label = QLabel(
            f"Free Memory:\t {human_readable_byte_size(self._available_virtual_memory())}, "
            f"({self._percentage_available_virtual_memory()}%)",
            self,
        )
        self.memory_group_box_layout.addWidget(self.free_memory_label)
        if self._available_virtual_memory() < 8000000000:
            self.free_memory_label.setStyleSheet("QLabel {color: red;}")
        elif self._available_virtual_memory() < 32000000000:
            self.free_memory_label.setStyleSheet("QLabel {color: orange;}")
        else:
            self.free_memory_label.setStyleSheet("QLabel {color: green;}")

        self.total_memory_label = QLabel(
            f"Total Memory:\t {human_readable_byte_size(self._total_virtual_memory())}",
            self,
        )
        self.memory_group_box_layout.addWidget(self.total_memory_label)
        if self._total_virtual_memory() < 8000000000:
            self.total_memory_label.setStyleSheet("QLabel {color: red;}")
        elif self._total_virtual_memory() < 32000000000:
            self.total_memory_label.setStyleSheet("QLabel {color: orange;}")
        else:
            self.total_memory_label.setStyleSheet("QLabel {color: green;}")

        # GPU summary
        self.gpu_group_box = QGroupBox("GPU Summary")
        self.gpu_group_box_layout = QVBoxLayout()
        self.gpu_group_box_layout.setSpacing(0)
        self.gpu_group_box_layout.setAlignment(Qt.AlignTop)
        self.gpu_group_box.setLayout(self.gpu_group_box_layout)

        cuda_gpu_name = self._cuda_gpu_name()


        self.cuda_gpu_label = QLabel(f"CUDA GPU: \t\t{cuda_gpu_name}", self)
        self.gpu_group_box_layout.addWidget(self.cuda_gpu_label)

        if cuda_gpu_name != "N/A":
            self.cuda_gpu_label.setStyleSheet("QLabel {color: green;}")
        else:
            self.cuda_gpu_label.setStyleSheet("QLabel {color: red;}")

        cuda_toolkit = self._cuda_toolkit_available()
        self.cudatoolkit_label = QLabel(
            f"CUDA Toolkit: \t\t{'present' if cuda_toolkit else 'absent'}", self
        )
        self.gpu_group_box_layout.addWidget(self.cudatoolkit_label)

        if cuda_toolkit:
            self.cudatoolkit_label.setStyleSheet("QLabel {color: green;}")
        else:
            self.cudatoolkit_label.setStyleSheet("QLabel {color: red;}")

        cuda_memory_free = self._cuda_free_mem()
        cuda_memory_total = self._cuda_total_mem()

        self.gpu_memory_free_label = QLabel(
            f"Free GPU Memory: \t{human_readable_byte_size(cuda_memory_free)}", self
        )
        self.gpu_group_box_layout.addWidget(self.gpu_memory_free_label)

        self.gpu_memory_total_label = QLabel(
            f"Total GPU Memory: \t{human_readable_byte_size(cuda_memory_total)}", self
        )
        self.gpu_group_box_layout.addWidget(self.gpu_memory_total_label)

        if cuda_memory_total == 0:
            self.gpu_memory_total_label.setStyleSheet("QLabel {color: red;}")
            self.gpu_memory_free_label.setStyleSheet("QLabel {color: red;}")
        else:
            if cuda_memory_total < 8000000000:
                self.gpu_memory_total_label.setStyleSheet("QLabel {color: orange;}")
            else:
                self.gpu_memory_total_label.setStyleSheet("QLabel {color: green;}")

            if cuda_memory_free / cuda_memory_total < 0.4:
                self.gpu_memory_free_label.setStyleSheet("QLabel {color: red;}")
            elif cuda_memory_free / cuda_memory_total < 0.8:
                self.gpu_memory_free_label.setStyleSheet("QLabel {color: orange;}")
            else:
                self.gpu_memory_free_label.setStyleSheet("QLabel {color: green;}")

        self.layout.addWidget(self.cpu_group_box)
        self.layout.addWidget(self.memory_group_box)
        self.layout.addWidget(self.gpu_group_box)

        self.setLayout(self.layout)

    def _cuda_total_mem(self):
        try:
            import numba
            return numba.cuda.current_context().get_memory_info().total
        except :
            return 0

    def _cuda_free_mem(self):
        try:
            import numba
            return numba.cuda.current_context().get_memory_info().free
        except:
            return 0

    def _cuda_toolkit_available(self):
        try:
            import numba
            return numba.cuda.cudadrv.nvvm.is_available()
        except:
            return False

    def _cuda_gpu_name(self):
        try:
            import numba
            from numba.cuda import CudaSupportError
            return numba.cuda.get_current_device().name.decode()
        except:
            return "N/A"


    def _total_virtual_memory(self):
        try:
            import psutil
            return psutil.virtual_memory().total
        except:
            lprint("Could not obtain total available virtual memory!")
            return -1

    def _percentage_available_virtual_memory(self):
        try:
            return round(
                100 * self._available_virtual_memory() / self._total_virtual_memory(),
                2)
        except:
            lprint("Could not obtain percentage of available virtual memory!")
            return 'unavailable'


    def _available_virtual_memory(self):
        try:
            import psutil
            return psutil.virtual_memory().available
        except:
            lprint("Could not obtain number of cores!")
            return 0


    def _cpu_load_values(self) -> Union[List, str]:
        try:
            import psutil
            return [(elem * 16) for elem in psutil.getloadavg()]
        except:
            # On some systems we can't determine the CPU load values, this should not crash Aydin...
            lprint("Could not determine the CPU load values!")
            return [-1, -1, -1]

    def _number_of_cores_int(self) -> int:
        try:
            return os.cpu_count() // 2
        except:
            # On some systems we can't deternmine the CPU speed, this should not crash Aydin...
            lprint("Could not obtain number of cores!")
            return -1

    def _number_of_cores(self) -> str:
        nb_cores = self._number_of_cores_int()
        return 'unavailable' if nb_cores < 0 else nb_cores

    def _cpu_freq(self) -> str:
        try:
            import psutil
            return f'{round(psutil.cpu_freq().current, 2)} Mhz'
        except:
            # On some systems we can't determine the CPU speed, this should not crash Aydin...
            lprint("Could not determine the CPU speed!")
            return 'unavailable'
