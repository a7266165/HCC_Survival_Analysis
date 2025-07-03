# src/utils/multiprocess_utils.py
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.config_utils import MultiprocessConfig
from typing import List, Any, Callable

logger = logging.getLogger(__name__)


def calculate_optimal_workers(total_tasks: int, config: MultiprocessConfig) -> int:
    """
    計算最佳的 worker 數量

    Args:
        total_tasks: 總任務數
        config: 多進程配置

    Returns:
        最佳 worker 數量
    """
    cpu_count = multiprocessing.cpu_count()

    if config.use_cpu_count:
        # 基於 CPU 數量計算
        available_cpus = max(cpu_count - config.reserve_cpus, 1)
        optimal_workers = min(available_cpus, total_tasks, config.max_workers)
    else:
        # 直接使用配置的數量
        optimal_workers = min(config.max_workers, total_tasks)

    logger.info(
        f"多進程配置：系統 CPU 數={cpu_count}, "
        f"可用 CPU 數={optimal_workers}, "
        f"總任務數={total_tasks}"
    )

    return optimal_workers


def run_parallel_tasks(
    task_function: Callable,
    task_args_list: List[tuple],
    config: MultiprocessConfig,
) -> List[Any]:
    """
    並行執行多個任務

    Args:
        task_function: 要執行的函數
        task_args_list: 任務參數列表
        task_name: 任務名稱（用於日誌）

    Returns:
        結果列表（保持原始順序）
    """
    total_tasks = len(task_args_list)

    # 如果不啟用多進程或只有一個任務，使用順序執行
    if not config.enabled or total_tasks <= 1:
        logger.info(f"使用順序處理模式執行 {total_tasks} 個實驗")
        results = []
        for i, args in enumerate(task_args_list):
            if config.show_progress:
                logger.info(f"進度: {i+1}/{total_tasks} ({(i+1)/total_tasks*100:.1f}%)")
            result = task_function(args)
            results.append(result)
        return results

    # 計算 worker 數量
    max_workers = calculate_optimal_workers(total_tasks, config)

    logger.info(f"使用 {max_workers} 個 CPU 核心並行處理 {total_tasks} 個實驗")

    # 使用並行處理
    results_dict = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務，記錄原始索引
        future_to_index = {
            executor.submit(task_function, args): (i, args)
            for i, args in enumerate(task_args_list)
        }

        # 處理完成的任務
        for future in as_completed(future_to_index):
            index, args = future_to_index[future]

            try:
                result = future.result()
                results_dict[index] = result
                completed += 1

                if config.show_progress:
                    logger.info(
                        f"進度: {completed}/{total_tasks} "
                        f"({completed/total_tasks*100:.1f}%) - "
                        f"完成索引 {index}"
                    )

            except Exception as exc:
                logger.error(f"實驗索引 {index} 執行失敗: {exc}")
                results_dict[index] = None
                completed += 1

    # 按原始順序重建結果列表
    results = [results_dict[i] for i in range(total_tasks)]

    logger.info(f"所有實驗執行完成，共 {total_tasks} 個")

    return results
