# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def initialize_task(config, env, init_sim=True):
    from learned_robot_placement.tasks.tiago_dual_whole_body_example import TiagoDualWBExampleTask
    from learned_robot_placement.tasks.tiago_dual_reaching import TiagoDualReachingTask
    from learned_robot_placement.tasks.tiago_dual_multiobj_fetching import TiagoDualMultiObjFetchingTask
    from learned_robot_placement.tasks.fetch_reaching import FetchReachingTask
    from learned_robot_placement.tasks.fetch_multiobj_fetching import FetchMultiObjFetchingTask

    # Mappings from strings to environments
    task_map = {
        "FetchReaching": FetchReachingTask,
        "FetchMultiObjFetching": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v11": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v12": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v13": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v14": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v15": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v21": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v22": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v23": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v24": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v25": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v31": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v32": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v33": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v34": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v35": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v41": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v42": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v43": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v44": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v45": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v51": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v52": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v53": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v54": FetchMultiObjFetchingTask,
        "FetchMultiObjFetching_v55": FetchMultiObjFetchingTask,
    }

    from .config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task