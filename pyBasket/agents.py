import numpy as np
from vimms.Agent import AbstractAgent


class DataDependantAction():
    def __init__(self):
        self.mz = None
        self.rt = None
        self.original_intensity = None
        self.scaled_intensity = None
        self.ms_level = 1
        self.idx = None

    def target(self, mz, rt, original_intensity, scaled_intensity, idx):
        self.mz = mz
        self.rt = rt
        self.original_intensity = original_intensity
        self.scaled_intensity = scaled_intensity
        self.idx = idx
        self.ms_level = 2


class DataDependantAcquisitionAgent(AbstractAgent):
    def __init__(self, isolation_window):
        super().__init__()
        self._initial_state()
        self.isolation_window = isolation_window

    def _initial_state(self):
        self.target_ms1()
        self.last_ms1_scan = None

    def target_ms1(self):
        self.dda_action = DataDependantAction()
        return self.dda_action

    def target_ms2(self, mz, rt, original_intensity, scaled_intensity, idx):
        self.dda_action = DataDependantAction()
        self.dda_action.target(mz, rt, original_intensity, scaled_intensity, idx)
        return self.dda_action

    def next_tasks(self, scan_to_process, controller, current_task_id):
        self.act(scan_to_process)
        if scan_to_process.ms_level == 1:
            self.last_ms1_scan = scan_to_process

        new_tasks = []
        if self.dda_action.ms_level == 1:
            scan_params = controller.get_ms1_scan_params()

        elif self.dda_action.ms_level == 2:
            precursor_scan_id = self.last_ms1_scan.scan_id
            scan_params = controller.get_ms2_scan_params(
                self.dda_action.mz, self.dda_action.original_intensity, precursor_scan_id,
                self.isolation_window, 0, 0)

        current_task_id += 1
        new_tasks.append(scan_params)
        next_processed_scan_id = current_task_id
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass

    def act(self, scan_to_process):
        pass

    def reset(self):
        self._initial_state()

    def _get_mzs_rt_intensities(self, scan_to_process):
        mzs = scan_to_process.mzs
        intensities = scan_to_process.intensities
        rt = scan_to_process.rt
        assert mzs.shape == intensities.shape
        return mzs, rt, intensities
