# 仓库级 Agent 执行计划：micrograd 三随机种子最小二分类实验

> 状态：`completed`
> 计划编号：`repo-plan-1df23982a7`
> 目标仓库：`$EVAL_ROOT/micrograd`

## 用户目标

为 micrograd 设计一个 CPU 可运行、固定三个随机种子的最小神经网络分类实验，输出 JSON 指标。

## Agent 的判断

仓库中现有入口 test/test_engine.py 依赖 PyTorch 且只做梯度正确性校验，不包含神经网络分类训练流程。为满足 CPU 可运行、固定三个随机种子并输出 JSON 指标的要求，新增一个无第三方依赖的最小 MLP 二分类训练评估脚本。脚本用合成二维高斯斑点数据集，构建 MLP(2, [4, 1])，用 hinge loss 和 SGD 训练后计算准确率与损失。实验矩阵为 seed=1、2、3 三个独立运行，每次运行把指标写入独立 JSON 文件供解析。

## Agent 阅读了什么

- `README.md`
- `micrograd/engine.py`
- `micrograd/nn.py`
- `setup.py`
- `test/test_engine.py`

## 拟执行实验

| Run | 用途 | Variant | Seed | 命令 | 超时 |
| --- | --- | --- | ---: | --- | ---: |
| `seed1` | 在固定种子 1 下训练并评估 MLP 二分类器，输出准确率和损失。 | seed=1; MLP(2,[4,1]); hinge loss; SGD | 1 | `python3 experiment_classification.py --seed 1 --metrics metrics_seed1.json` | 120s |
| `seed2` | 在固定种子 2 下训练并评估 MLP 二分类器，输出准确率和损失。 | seed=2; MLP(2,[4,1]); hinge loss; SGD | 2 | `python3 experiment_classification.py --seed 2 --metrics metrics_seed2.json` | 120s |
| `seed3` | 在固定种子 3 下训练并评估 MLP 二分类器，输出准确率和损失。 | seed=3; MLP(2,[4,1]); hinge loss; SGD | 3 | `python3 experiment_classification.py --seed 3 --metrics metrics_seed3.json` | 120s |

## 拟修改代码

### `experiment_classification.py`

提供可运行的分类实验入口：固定随机种子、生成合成数据、训练 MLP、输出 JSON 指标文件。

```diff
--- a/experiment_classification.py
+++ b/experiment_classification.py
@@ -0,0 +1,63 @@
+import argparse
+import json
+import random
+from micrograd.engine import Value
+from micrograd.nn import MLP
+
+
+def make_dataset(seed):
+    rng = random.Random(seed)
+    xs = []
+    ys = []
+    for _ in range(64):
+        cls = rng.randint(0, 1)
+        x = [rng.gauss(-0.6, 0.6) if cls == 0 else rng.gauss(0.6, 0.6) for _ in range(2)]
+        xs.append(x)
+        ys.append(1.0 if cls == 1 else -1.0)
+    return xs, ys
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument('--seed', type=int, required=True)
+    parser.add_argument('--metrics', required=True)
+    args = parser.parse_args()
+
+    seed = args.seed
+    random.seed(seed)
+    xs, ys = make_dataset(seed)
+    model = MLP(2, [4, 1])
+
+    lr = 0.1
+    epochs = 15
+    for _ in range(epochs):
+        loss = Value(0.0)
+        for x, y in zip(xs, ys):
+            pred = model(x)
+            loss = loss + (1 - y * pred).relu()
+        loss = loss / len(xs)
+        model.zero_grad()
+        loss.backward()
+        for p in model.parameters():
+            p.data -= lr * p.grad
+
+    correct = 0
+    total_loss = 0.0
+    for x, y in zip(xs, ys):
+        pred = model(x).data
+        total_loss += max(0.0, 1 - y * pred)
+        if y * pred > 0:
+            correct += 1
+
+    metrics = {
+        'seed': seed,
+        'accuracy': correct / len(xs),
+        'loss': total_loss / len(xs),
+    }
+    with open(args.metrics, 'w') as f:
+        json.dump(metrics, f)
+    print(json.dumps(metrics))
+
+
+if __name__ == '__main__':
+    main()
```

## 审批边界

审批前不会写入代码，也不会执行任何命令。
执行时不使用 Shell，不自动安装依赖，不允许内联 Python 和目录越界。
