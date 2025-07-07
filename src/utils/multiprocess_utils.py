# src/utils/multiprocess_utils.py
import logging
import multiprocessing
from multiprocessing import get_context
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
    並行執行多個任務，特別處理 CoxPHFitter

    Args:
        task_function: 要執行的函數
        task_args_list: 任務參數列表
        config: 多進程配置

    Returns:
        結果列表（保持原始順序）
    """
    total_tasks = len(task_args_list)

    # 分離 Cox 和非 Cox 任務
    cox_tasks = []
    other_tasks = []

    for i, args in enumerate(task_args_list):
        # 檢查是否為 CoxPHFitter（model_type 在 args 的第 4 個位置）
        if len(args) > 4 and args[4] == "CoxPHFitter":
            cox_tasks.append((i, args))
        else:
            other_tasks.append((i, args))

    logger.info(
        f"任務分配：CoxPHFitter 任務 {len(cox_tasks)} 個（順序執行），"
        f"其他任務 {len(other_tasks)} 個（並行執行）"
    )

    # 初始化結果字典
    results_dict = {}

    # 1. 順序執行 Cox 任務
    if cox_tasks:
        logger.info(f"開始順序執行 {len(cox_tasks)} 個 CoxPHFitter 任務")
        for task_index, (original_index, args) in enumerate(cox_tasks):
            try:
                if config.show_progress:
                    logger.info(
                        f"Cox 進度: {task_index + 1}/{len(cox_tasks)} "
                        f"({(task_index + 1) / len(cox_tasks) * 100:.1f}%)"
                    )
                result = task_function(args)
                results_dict[original_index] = result
            except Exception as e:
                logger.error(f"CoxPHFitter 任務索引 {original_index} 執行失敗: {e}")
                results_dict[original_index] = None

    # 2. 並行執行其他任務
    if other_tasks:
        # 如果不啟用多進程或只有一個任務，使用順序執行
        if not config.enabled or len(other_tasks) <= 1:
            logger.info(f"使用順序處理模式執行 {len(other_tasks)} 個其他任務")
            for task_index, (original_index, args) in enumerate(other_tasks):
                try:
                    if config.show_progress:
                        logger.info(
                            f"其他任務進度: {task_index + 1}/{len(other_tasks)} "
                            f"({(task_index + 1) / len(other_tasks) * 100:.1f}%)"
                        )
                    result = task_function(args)
                    results_dict[original_index] = result
                except Exception as e:
                    logger.error(f"任務索引 {original_index} 執行失敗: {e}")
                    results_dict[original_index] = None
        else:
            # 使用並行處理
            max_workers = calculate_optimal_workers(len(other_tasks), config)
            logger.info(
                f"使用 {max_workers} 個 CPU 核心並行處理 {len(other_tasks)} 個其他任務"
            )

            completed = 0
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有非 Cox 任務
                future_to_index = {
                    executor.submit(task_function, args): (original_index, args)
                    for original_index, args in other_tasks
                }

                # 處理完成的任務
                for future in as_completed(future_to_index):
                    original_index, args = future_to_index[future]

                    try:
                        result = future.result()
                        results_dict[original_index] = result
                        completed += 1

                        if config.show_progress:
                            logger.info(
                                f"並行任務進度: {completed}/{len(other_tasks)} "
                                f"({completed / len(other_tasks) * 100:.1f}%) - "
                                f"完成索引 {original_index}"
                            )

                    except Exception as exc:
                        logger.error(f"實驗索引 {original_index} 執行失敗: {exc}")
                        results_dict[original_index] = None
                        completed += 1

    # 按原始順序重建結果列表
    results = [results_dict[i] for i in range(total_tasks)]

    logger.info(
        f"所有任務執行完成，共 {total_tasks} 個"
        f"（Cox: {len(cox_tasks)}，其他: {len(other_tasks)}）"
    )

    return results
