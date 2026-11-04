# 실험 결과 요약

2026-10-27 현재 저장된 결과 기준. `benchmark/splits.json`의 test split은 11개 코호트, 386개 샘플, 11,671개 step이다. 정의가 바뀐 14개 step의 test task 328개만 11개 모델에서 다시 실행하고, 나머지 11,343개 예측은 기존 출력을 유지했다. 전체 점수는 **코호트별 점수를 먼저 구한 뒤 11개 코호트를 동일 비중으로 평균**했다. F1은 macro-F1, BalAcc는 balanced accuracy다. Hull IoU도 코호트별 유효 step 평균을 구한 뒤 동일 비중으로 평균했다. 모든 점수는 높을수록 좋다.

## 전체 결과

| 방법 | GT F1 | GT BalAcc | GT Hull IoU | rect GT F1 | rect GT BalAcc | rect GT Hull IoU |
|---|---:|---:|---:|---:|---:|---:|
| rect-1 GT-informed reference | 0.908 | 0.974 | 0.950 | 1.000 | 1.000 | 1.000 |
| flowDensity + rule (LLM 없음) | 0.578 | 0.679 | 0.516 | 0.574 | 0.675 | 0.509 |
| C2S grid · GPT-5.4 | 0.589 | 0.689 | 0.531 | 0.585 | 0.680 | 0.525 |
| C2S grid · GPT-6 luna | 0.626 | 0.722 | 0.572 | 0.621 | 0.715 | 0.566 |
| C2S grid · GPT-6 luna medium | 0.626 | 0.722 | 0.572 | 0.622 | 0.714 | 0.565 |
| C2S grid · GPT-6 sol medium | 0.641 | 0.735 | 0.591 | 0.636 | 0.725 | 0.584 |
| C2S FlowSOM · GPT-5.4 | 0.530 | 0.644 | 0.451 | 0.525 | 0.633 | 0.444 |
| C2S FlowSOM · GPT-6 luna | 0.604 | 0.707 | 0.543 | 0.599 | 0.696 | 0.533 |
| C2S FlowSOM · GPT-6 luna medium | 0.601 | 0.704 | 0.539 | 0.596 | 0.693 | 0.531 |
| C2S FlowSOM · GPT-6 sol medium | 0.640 | 0.737 | 0.584 | 0.634 | 0.725 | 0.575 |
| LLM Gate · GPT-5.4 (논문 실행) | **0.676** | **0.768** | 0.608 | — | — | — |
| LLM Gate · GPT-5.4 (논문 게이트 재적용) | 0.676 | 0.768 | 0.608 | 0.675 | 0.762 | 0.604 |
| LLM Gate · GPT-5.4 (현재 정의) | 0.634 | 0.740 | 0.559 | 0.633 | 0.736 | 0.554 |
| LLM Gate · GPT-6 luna 기본값 | 0.654 | 0.748 | 0.606 | 0.653 | 0.741 | 0.602 |
| LLM Gate · GPT-6 luna none | 0.595 | 0.705 | 0.545 | 0.595 | 0.701 | 0.542 |
| LLM Gate · GPT-6 luna low | 0.638 | 0.740 | 0.587 | 0.637 | 0.733 | 0.583 |
| LLM Gate · GPT-6 luna medium | 0.651 | 0.747 | 0.602 | 0.651 | 0.740 | 0.600 |
| LLM Gate · GPT-6 luna high | 0.656 | 0.746 | 0.607 | 0.656 | 0.739 | 0.604 |
| LLM Gate · GPT-6 sol medium | 0.671 | 0.747 | 0.617 | 0.670 | 0.739 | 0.614 |

`GT`는 원래 세포 주석이고, 이 표의 `rect GT`는 rect-1의 직사각형 라벨이다. rect-1 행은 모델 성능이나 수학적으로 증명된 상한이 아니라 GT를 사용해 탐색한 최선 발견 해다. 탐색은 전역 최적성을 보장하지 않고, 실제 LLM-Gate는 범주별 여러 직사각형을 허용한다. flowDensity 행은 저장된 baseline 예측을 rect-1과 [직접 평가](../jobs/eval_flowdensity_rule_rect1.sh)하고, GT/rect GT 모두 [11개 코호트 동일 비중으로 집계](../results/flowdensity_rule/rect_gt_summary.json)했다. 논문 GPT-5.4 원본 실행에는 rect GT 평가 결과가 없어 빈칸으로 두었다. **논문 게이트 재적용** 행은 이전 GPT-5.4 실행에 저장된 게이트 좌표를 현재 benchmark의 세포 집합에 다시 적용해 GT와 rect GT를 함께 평가한 결과다. 현재 정의 모델 행은 [11개 방법의 동일 split 집계](../results/vs_gt_rect1_summary.md)에서 가져왔다. 새 C2S medium 네 행은 [별도 전체 test 집계](../results/c2s_medium_full_summary.md)에서 가져왔으며, 기존 GPT-6 luna 기본 설정 행과 구분한다.

## Rect-2 GT-informed screen (2026-10-27)

저장된 rect-1 게이트에서 출발해 각 범주에 겹치지 않는 두 번째 직사각형을 최대 1개 추가했다. 64-bin 후보를 탐색하고, step macro-F1이 최소 0.01 개선되며 추가 직사각형이 해당 범주 GT 세포의 최소 5%이자 최소 10개를 회수할 때만 채택했다. **11,671개 중 104개 step**에 두 번째 직사각형이 추가됐다. 여기서 “필요한 step”은 이 기준을 통과한 *기존 rect-1 기준의 예비 후보*를 뜻하며, 수학적 필요성이나 전역 최적해를 증명한 결과는 아니다. 기존 rect-1 탐색은 전체 macro-F1을 점수로 사용했지만, 초기값을 label 순서대로 만들고 한 번에 한 변만 움직여 국소해에 머물 수 있었다. 이 때문에 104개를 진정한 rect-2 필요 사례로 해석해서는 안 된다. 세포 1개만 추가해 점수가 크게 오르는 극소 부모 집합 사례는 10세포 기준으로 제외했다.

| GT-informed reference | GT F1 | GT BalAcc | GT Hull IoU |
|---|---:|---:|---:|
| rect-1 | 0.9079 | 0.9744 | 0.9499 |
| rect-2 screen (최소 10세포) | 0.9095 | 0.9750 | 0.9535 |

점수는 동일 test split의 11개 코호트 동일 비중 평균이다. 가장 큰 반복 사례는 Bjornson `step_08`의 DCs로, GT 세포가 2D 공간의 좌상단과 우하단에 나뉘어 있어 두 번째 직사각형이 큰 개선을 준다. 전체 104개 step의 표본명, step, 대상 범주, 전후 F1, 추가 게이트 좌표는 [후보 목록](../results/rect2_gt_min10/changed_steps.csv)에 있다. [코호트별 요약](../results/rect2_gt_min10/summary.md), [F1·BA·Hull IoU 비교](../results/rect2_gt_min10/eval_comparison.md), [상위 사례 시각화](../results/rect2_gt_min10/top_six_gate_review.png)도 저장했다. 새 세포별 rect-GT는 `results/rect2_gt_min10/{cohort}/{sample}/{step}/prediction.json`에 있으며, 변경되지 않은 step은 rect-1 결과를 참조한다. 변경된 104개 step은 게이트 좌표로 재생성한 라벨과 저장 라벨 및 F1의 일치를 확인했다. 이 rect-2 reference는 기존 표의 rect-1 기준 모델 평가를 자동으로 대체하지 않는다.

2026-10-27 사용자 검토: **Bjornson `step_08`은 rect-2로 승인**. [후보 게이트 감사](../results/rect2_gt_min10/true_second_audit.md)에서 104개 모두 해당 범주의 rect-1 게이트가 이미 존재해, 실제 두 번째 직사각형을 추가한 경우임을 확인했다. 다른 코호트의 후보 step은 아직 승인되지 않았다.

기존 rect-2 후보 104개를 [공동 rect-1 탐색](../src/baselines/rect_oracle/fit_rect1_joint.py)으로 재평가했다. 이 탐색은 한두 label의 직사각형 전체를 비겹침 제약 아래 재배치하고, 모든 범주와 Unassigned의 step macro-F1을 기준으로 선택한다. 기존 rect-1 해도 시작 후보로 보존하므로 F1이 낮아진 해는 채택하지 않는다. 단, 이 역시 전역 최적성의 증명은 아니다. [104개 후보 재평가 요약](../results/rect1_joint_candidates/summary.md)에 따르면 **57개에서 rect-1 F1이 개선됐고, 36개는 기존 rect-2와의 F1 차이가 0.01 미만으로 줄었다**. 나머지 68개는 기존 rect-2가 최소 0.01 앞선다. 이는 *기존 rect-2 해와의 비교*이며, 공동 rect-1 해에서 rect-2를 다시 최적화한 결과는 아니다. [전체 test step 재적합](../results/rect1_joint_full/summary.md)도 11,671개 모두 완료됐다. **1,017개 step에서 F1이 개선됐고, 11개 코호트 동일 가중 step macro-F1은 기존 rect-1의 0.9079에서 공동 rect-1의 0.9109로 상승했다.** 위쪽 전체 결과 표는 아직 원래 rect-1과 그에 대한 모델 평가값을 표시한다.

## 코호트별 F1 (rect GT 기준)

| 코호트 | flowDensity + rule | C2S grid 5.4 | C2S grid luna | C2S FlowSOM 5.4 | C2S FlowSOM luna | Gate 5.4 논문 게이트 재적용 | Gate 5.4 현재 정의 | Gate luna 기본 | Gate luna none | Gate luna low | Gate luna medium | Gate luna high | Gate sol medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acute2020 | 0.471 | 0.553 | 0.572 | 0.597 | 0.658 | 0.651 | 0.637 | 0.646 | 0.588 | 0.643 | 0.642 | 0.643 | 0.656 |
| Acute2021 | 0.497 | 0.575 | 0.583 | 0.602 | 0.659 | 0.643 | 0.621 | 0.632 | 0.575 | 0.625 | 0.631 | 0.633 | 0.660 |
| Bjornson | 0.700 | 0.715 | 0.746 | 0.639 | 0.719 | 0.802 | 0.783 | 0.741 | 0.681 | 0.749 | 0.741 | 0.745 | 0.781 |
| Vaccine | 0.484 | 0.544 | 0.569 | 0.589 | 0.669 | 0.668 | 0.648 | 0.659 | 0.594 | 0.654 | 0.658 | 0.662 | 0.676 |
| FR-FCM-Z74D_hc | 0.591 | 0.618 | 0.620 | 0.554 | 0.572 | 0.657 | 0.620 | 0.632 | 0.560 | 0.610 | 0.626 | 0.638 | 0.640 |
| FR-FCM-Z74D_tissue | 0.463 | 0.507 | 0.505 | 0.477 | 0.519 | 0.554 | 0.517 | 0.536 | 0.483 | 0.530 | 0.538 | 0.530 | 0.557 |
| FRDR_covid19 | 0.594 | 0.625 | 0.654 | 0.614 | 0.685 | 0.661 | 0.646 | 0.670 | 0.623 | 0.665 | 0.669 | 0.676 | 0.697 |
| Lyoplate_DC | 0.750 | 0.646 | 0.754 | 0.367 | 0.516 | 0.708 | 0.699 | 0.680 | 0.646 | 0.670 | 0.680 | 0.681 | 0.695 |
| Lyoplate_bcell | 0.611 | 0.591 | 0.629 | 0.527 | 0.611 | 0.713 | 0.610 | 0.693 | 0.576 | 0.642 | 0.684 | 0.697 | 0.710 |
| Lyoplate_tcell | 0.617 | 0.479 | 0.596 | 0.290 | 0.392 | 0.650 | 0.535 | 0.582 | 0.520 | 0.514 | 0.577 | 0.587 | 0.624 |
| Lyoplate_treg | 0.535 | 0.576 | 0.608 | 0.524 | 0.586 | 0.721 | 0.649 | 0.715 | 0.693 | 0.701 | 0.710 | 0.720 | 0.678 |

코호트별 GT Hull IoU와 현재 정의 모델의 다른 세부 수치는 [원본 비교표](../results/vs_gt_rect1_summary.md)에 있다. flowDensity의 코호트별 GT/rect GT 점수는 [baseline 집계](../results/flowdensity_rule/rect_gt_summary.json), 논문 게이트 재적용의 코호트별 점수는 [재적용 집계](../results/gate_gpt-5.4_paper_gates_current_benchmark/summary.json)에 있다. 위 표의 코호트 값은 각 코호트 내부 step 평균이다.

## 해석 및 비교 조건

- GT F1 최고는 논문 GPT-5.4 실행(0.676)이다. 현재 정의 실행만 보면 Gate · GPT-6 sol medium(0.671)이 가장 높고, GT Hull IoU도 0.617로 가장 높다.
- Gate · GPT-6 luna는 `none`에서 F1 0.595, `low`에서 0.638, `medium`에서 0.651, `high`에서 0.656이다. 기본값 실행은 0.654이다.
- 논문 GPT-5.4 실행은 이전 `notes.yaml` 문구를 사용했다. 현재 정의의 GPT-5.4와 다른 프롬프트 조건이다. [대응 step 비교](../results/gate_full_test_summary.md)는 공통 11,664 step 기준이다.
- flowDensity baseline에는 427개 deGate 오류 step이 포함되어 있으며, 실패 step도 점수에 반영됐다. 나머지 전체 비교 행의 원본 집계는 `results/*/predictions/<cohort>/eval_summary.json`에 있다.

## 논문 GPT-5.4 수치와의 차이

논문 수치는 이전 GPT-5.4 실행의 동일한 output에서 **11개 코호트 평균을 동일 비중으로** 계산한 값이다. 이 방식으로 저장된 `eval_summary.json`을 다시 집계하면 제공된 논문 수치와 반올림까지 일치한다 ([집계 작업](../jobs/compare_paper_gpt54.sh)).

| GPT-5.4 결과 | 평가 step | GT F1 | GT BalAcc | GT Hull IoU | rect GT F1 | rect GT BalAcc | rect GT Hull IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 논문 표 / 이전 실행 (`gate_gpt-5.4_full_old`) | 11,665 | 0.676 | 0.768 | 0.608 | — | — | — |
| 이전 게이트를 현재 benchmark에 재적용 | 11,664 | 0.676 | 0.768 | 0.608 | 0.675 | 0.762 | 0.604 |
| 현재 정의 실행 (`gate_gpt-5.4_full`) | 11,671 | 0.634 | 0.740 | 0.559 | 0.633 | 0.736 | 0.554 |

모든 전체 점수는 각 코호트가 1/11의 비중을 갖는다. 이전 실행 디렉터리에 남아 있는 YoungOld 코호트는 이 11개 코호트 비교에서 제외했다. 코호트별 평가 step 수는 합계만 보고하며 전체 평균의 가중치로 사용하지 않는다.

이전 실행의 세포별 예측 라벨은 현재 benchmark 및 rect-1 oracle의 부모 세포 수와 일부 step에서 달라 직접 rect GT 평가에 사용할 수 없다. 따라서 저장된 이전 게이트 좌표(`pred.json`)를 현재 benchmark에 재적용하고, 동일한 재적용 예측에 대해 GT와 rect GT를 함께 평가했다 ([평가 작업](../jobs/eval_paper_gates_current_rect1.sh), [11개 코호트 집계](../jobs/summarize_paper_gates_current_rect1.sh), [결과](../results/gate_gpt-5.4_paper_gates_current_benchmark/summary.json)). Acute2021의 한 step은 현재 `task.json`이 없어 제외되어 11,664 step이다. 재적용 GT 점수는 원래 논문 점수와 소수 셋째 자리까지 일치하지만, rect GT 수치는 **원래 11,665 step 논문 실행의 직접 평가값이 아니다**.

이전 실행과 **현재 정의 실행** 사이의 성능 차이는 별개다. 현재 결과는 `benchmark/notes.yaml`의 `definition`만 프롬프트에 넣은 전체 실행에서, 정의가 수정된 step만 재호출해 합친 것이다 ([원래 실행 스크립트](../jobs/run_gate_full_test.sh), [선별 재실행 작업](../jobs/run_definition_rerun_20261027.sh)). 이전 note에는 상대 위치 등의 힌트가 있었고, 2026-10-20의 259개 대응 step 진단에서 이전 note를 다시 넣자 F1이 0.621에서 0.660으로 올랐다 ([진단 로그](../jobs/logs/diag_plots.49738149.out)). 이 결과는 두 실행의 차이를 해석할 때만 관련된다.
