# 실험 결과 요약

2026-10-27 현재 저장된 결과 기준. `benchmark/splits.json`의 test split은 11개 코호트, 386개 샘플, 11,671개 step이다. 전체 점수는 **코호트별 점수를 먼저 구한 뒤 11개 코호트를 동일 비중으로 평균**했다. F1은 macro-F1, BalAcc는 balanced accuracy다. Hull IoU도 코호트별 유효 step 평균을 구한 뒤 동일 비중으로 평균했다. 모든 점수는 높을수록 좋다.

## 전체 결과

| 방법 | GT F1 | GT BalAcc | GT Hull IoU | rect GT F1 | rect GT BalAcc | rect GT Hull IoU |
|---|---:|---:|---:|---:|---:|---:|
| rect-1 oracle (상한 참고) | 0.908 | 0.974 | 0.950 | 1.000 | 1.000 | 1.000 |
| flowDensity + rule (LLM 없음) | 0.578 | 0.679 | 0.516 | 0.574 | 0.675 | 0.509 |
| C2S grid · GPT-5.4 | 0.589 | 0.690 | 0.533 | 0.584 | 0.680 | 0.525 |
| C2S grid · GPT-6 luna | 0.625 | 0.722 | 0.571 | 0.620 | 0.714 | 0.564 |
| C2S FlowSOM · GPT-5.4 | 0.529 | 0.643 | 0.448 | 0.524 | 0.631 | 0.440 |
| C2S FlowSOM · GPT-6 luna | 0.602 | 0.705 | 0.540 | 0.596 | 0.694 | 0.531 |
| LLM Gate · GPT-5.4 (논문 실행) | **0.676** | **0.768** | 0.608 | — | — | — |
| LLM Gate · GPT-5.4 (논문 게이트 재적용) | 0.676 | 0.768 | 0.608 | 0.675 | 0.762 | 0.604 |
| LLM Gate · GPT-5.4 (현재 정의) | 0.636 | 0.742 | 0.564 | 0.635 | 0.738 | 0.558 |
| LLM Gate · GPT-6 luna 기본값 | 0.654 | 0.749 | 0.608 | 0.653 | 0.741 | 0.602 |
| LLM Gate · GPT-6 luna none | 0.596 | 0.705 | 0.546 | 0.595 | 0.702 | 0.542 |
| LLM Gate · GPT-6 luna low | 0.638 | 0.740 | 0.589 | 0.636 | 0.733 | 0.583 |
| LLM Gate · GPT-6 luna medium | 0.653 | 0.749 | 0.604 | 0.650 | 0.740 | 0.599 |
| LLM Gate · GPT-6 luna high | 0.657 | 0.747 | 0.609 | 0.656 | 0.739 | 0.603 |
| LLM Gate · GPT-6 sol medium | 0.670 | 0.745 | **0.616** | **0.669** | 0.738 | **0.612** |

`GT`는 원래 세포 주석이고, `rect GT`는 rect-1 oracle의 직사각형 라벨이다. Oracle 행은 모델 성능이 아닌 직사각형 표현의 참고 상한이다. flowDensity 행은 저장된 baseline 예측을 rect-1 oracle과 [직접 평가](../jobs/eval_flowdensity_rule_rect1.sh)하고, GT/rect GT 모두 [11개 코호트 동일 비중으로 집계](../results/flowdensity_rule/rect_gt_summary.json)했다. 논문 GPT-5.4 원본 실행에는 rect GT 평가 결과가 없어 빈칸으로 두었다. **논문 게이트 재적용** 행은 이전 GPT-5.4 실행에 저장된 게이트 좌표를 현재 benchmark의 세포 집합에 다시 적용해 GT와 rect GT를 함께 평가한 결과다. 현재 정의 모델 행은 [11개 방법의 동일 split 집계](../results/vs_gt_rect1_summary.md)에서 가져왔다.

## 코호트별 F1 (rect GT 기준)

| 코호트 | flowDensity + rule | C2S grid 5.4 | C2S grid luna | C2S FlowSOM 5.4 | C2S FlowSOM luna | Gate 5.4 논문 게이트 재적용 | Gate 5.4 현재 정의 | Gate luna 기본 | Gate luna none | Gate luna low | Gate luna medium | Gate luna high | Gate sol medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acute2020 | 0.471 | 0.550 | 0.568 | 0.594 | 0.655 | 0.651 | 0.637 | 0.641 | 0.583 | 0.639 | 0.638 | 0.639 | 0.657 |
| Acute2021 | 0.497 | 0.572 | 0.579 | 0.598 | 0.654 | 0.643 | 0.621 | 0.628 | 0.571 | 0.621 | 0.628 | 0.629 | 0.660 |
| Bjornson | 0.700 | 0.715 | 0.746 | 0.639 | 0.719 | 0.802 | 0.783 | 0.741 | 0.681 | 0.749 | 0.741 | 0.745 | 0.781 |
| Vaccine | 0.484 | 0.541 | 0.565 | 0.587 | 0.666 | 0.668 | 0.648 | 0.657 | 0.592 | 0.652 | 0.656 | 0.660 | 0.676 |
| FR-FCM-Z74D_hc | 0.591 | 0.623 | 0.616 | 0.550 | 0.564 | 0.657 | 0.627 | 0.628 | 0.569 | 0.609 | 0.628 | 0.643 | 0.627 |
| FR-FCM-Z74D_tissue | 0.463 | 0.506 | 0.503 | 0.470 | 0.510 | 0.554 | 0.526 | 0.545 | 0.496 | 0.531 | 0.543 | 0.534 | 0.557 |
| FRDR_covid19 | 0.594 | 0.625 | 0.654 | 0.614 | 0.685 | 0.661 | 0.646 | 0.670 | 0.623 | 0.665 | 0.669 | 0.676 | 0.697 |
| Lyoplate_DC | 0.750 | 0.646 | 0.754 | 0.367 | 0.516 | 0.708 | 0.699 | 0.680 | 0.646 | 0.670 | 0.680 | 0.681 | 0.695 |
| Lyoplate_bcell | 0.611 | 0.591 | 0.629 | 0.527 | 0.611 | 0.713 | 0.610 | 0.693 | 0.576 | 0.642 | 0.684 | 0.697 | 0.710 |
| Lyoplate_tcell | 0.617 | 0.479 | 0.596 | 0.290 | 0.392 | 0.650 | 0.535 | 0.582 | 0.520 | 0.514 | 0.577 | 0.587 | 0.624 |
| Lyoplate_treg | 0.535 | 0.576 | 0.608 | 0.524 | 0.586 | 0.721 | 0.649 | 0.715 | 0.693 | 0.701 | 0.710 | 0.720 | 0.678 |

코호트별 GT Hull IoU와 현재 정의 모델의 다른 세부 수치는 [원본 비교표](../results/vs_gt_rect1_summary.md)에 있다. flowDensity의 코호트별 GT/rect GT 점수는 [baseline 집계](../results/flowdensity_rule/rect_gt_summary.json), 논문 게이트 재적용의 코호트별 점수는 [재적용 집계](../results/gate_gpt-5.4_paper_gates_current_benchmark/summary.json)에 있다. 위 표의 코호트 값은 각 코호트 내부 step 평균이다.

## 해석 및 비교 조건

- GT F1 최고는 논문 GPT-5.4 실행(0.676)이다. 현재 정의 실행만 보면 Gate · GPT-6 sol medium(0.670)이 가장 높고, GT Hull IoU도 0.616으로 가장 높다.
- Gate · GPT-6 luna는 `none`에서 F1 0.596, `low`에서 0.638, `medium`에서 0.653, `high`에서 0.657이다. 기본값 실행은 0.654이다.
- 논문 GPT-5.4 실행은 이전 `notes.yaml` 문구를 사용했다. 현재 정의의 GPT-5.4와 다른 프롬프트 조건이다. [대응 step 비교](../results/gate_full_test_summary.md)는 공통 11,664 step 기준이다.
- flowDensity baseline에는 427개 deGate 오류 step이 포함되어 있으며, 실패 step도 점수에 반영됐다. 나머지 전체 비교 행의 원본 집계는 `results/*/predictions/<cohort>/eval_summary.json`에 있다.

## 논문 GPT-5.4 수치와의 차이

논문 수치는 이전 GPT-5.4 실행의 동일한 output에서 **11개 코호트 평균을 동일 비중으로** 계산한 값이다. 이 방식으로 저장된 `eval_summary.json`을 다시 집계하면 제공된 논문 수치와 반올림까지 일치한다 ([집계 작업](../jobs/compare_paper_gpt54.sh)).

| GPT-5.4 결과 | 평가 step | GT F1 | GT BalAcc | GT Hull IoU | rect GT F1 | rect GT BalAcc | rect GT Hull IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 논문 표 / 이전 실행 (`gate_gpt-5.4_full_old`) | 11,665 | 0.676 | 0.768 | 0.608 | — | — | — |
| 이전 게이트를 현재 benchmark에 재적용 | 11,664 | 0.676 | 0.768 | 0.608 | 0.675 | 0.762 | 0.604 |
| 현재 정의 실행 (`gate_gpt-5.4_full`) | 11,671 | 0.636 | 0.742 | 0.564 | 0.635 | 0.738 | 0.558 |

모든 전체 점수는 각 코호트가 1/11의 비중을 갖는다. 이전 실행 디렉터리에 남아 있는 YoungOld 코호트는 이 11개 코호트 비교에서 제외했다. 코호트별 평가 step 수는 합계만 보고하며 전체 평균의 가중치로 사용하지 않는다.

이전 실행의 세포별 예측 라벨은 현재 benchmark 및 rect-1 oracle의 부모 세포 수와 일부 step에서 달라 직접 rect GT 평가에 사용할 수 없다. 따라서 저장된 이전 게이트 좌표(`pred.json`)를 현재 benchmark에 재적용하고, 동일한 재적용 예측에 대해 GT와 rect GT를 함께 평가했다 ([평가 작업](../jobs/eval_paper_gates_current_rect1.sh), [11개 코호트 집계](../jobs/summarize_paper_gates_current_rect1.sh), [결과](../results/gate_gpt-5.4_paper_gates_current_benchmark/summary.json)). Acute2021의 한 step은 현재 `task.json`이 없어 제외되어 11,664 step이다. 재적용 GT 점수는 원래 논문 점수와 소수 셋째 자리까지 일치하지만, rect GT 수치는 **원래 11,665 step 논문 실행의 직접 평가값이 아니다**.

이전 실행과 **현재 새 실행** 사이의 성능 차이는 별개다. 현재 실행은 `benchmark/notes.yaml`의 `definition`만 프롬프트에 넣는다 ([실행 스크립트](../jobs/run_gate_full_test.sh)). 이전 note에는 상대 위치 등의 힌트가 있었고, 259개 대응 step 진단에서 이전 note를 다시 넣자 F1이 0.621에서 0.660으로 올랐다 ([진단 로그](../jobs/logs/diag_plots.49738149.out)). 이 결과는 두 실행의 차이를 해석할 때만 관련된다.
