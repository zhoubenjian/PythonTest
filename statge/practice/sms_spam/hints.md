# 提示卡

规则：卡住超过 20 分钟才看；从「阶段 1」开始看，还卡再往下。**做完再看答案等于白练。**

---

## 1 · load.py

- **阶段 1（方向）**：`pandas` 读分隔符不是逗号的文件，`read_csv` 的 `sep` 参数可以指定任意分隔符；没有表头要告诉它 `header=None`，列名用 `names=` 传。
- **阶段 2（API）**：`pd.read_csv(path, sep='\t', header=None, names=['label', 'text'])`

---

## 2 · split.py

- **阶段 1（方向）**：`train_test_split` 有一个参数专门处理「保持比例」——不平衡数据全靠它。分两刀切：先切 train / rest，再把 rest 对半。
- **阶段 2（API）**：

```python
train_df, rest = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(rest, test_size=0.5, stratify=rest['label'], random_state=42)
```

（想清楚 stratify 传的应该是哪一列再抄。）

---

## 3 · fit.py

- **阶段 1（方向）**：`TfidfVectorizer(max_features=5000)`；朴素贝叶斯（`MultinomialNB`）在稀疏向量上训练很快，适合当第一个基线；`LogisticRegression` 当第二个。分数用 `f1_score(y_true, y_pred, average='macro')`。
- **阶段 2（API）**：不需要 Pipeline，手动三步即可：`vectorizer.fit_transform(train_texts)` → `vectorizer.transform(val_texts)` → `model.fit(X_train, y_train)`。选 best 用 `max()` 对比两个分数，别写死名字。

---

## 4 · evaluate.py

- **阶段 1（方向）**：混淆矩阵用 `confusion_matrix`；标签按字母序排列，即行/列顺序是 `['ham', 'spam']`。错分样本定位：`y_pred != y_true` 的布尔索引回查 `test_df`。
- **阶段 2（API）**：

```python
mask = y_pred != y_true
wrong = test_df.loc[mask]   # 再和 y_pred 对齐打印
```

（对齐方式自己想：`y_pred[mask]` 和 `test_df.loc[mask]` 怎么逐行配对打印。）

---

## 5 · api/main.py + test_api.py

- **阶段 1（方向）**：结构可以参考 `text_classification/rebuilt/api/` 的文件组织（schemas 放哪、异常处理器怎么挂），但代码自己写。422 用 `HTTPException(status_code=422, detail=...)` 加一个自定义异常处理器转成 `{code, message}` 外形。`confidence` 来自 `model.predict_proba`，**列顺序由 `model.classes_` 决定**，别想当然。
- **阶段 2（API）**：

```python
prob = model.predict_proba(X)[0]        # 顺序对应 model.classes_
label_idx = int(prob.argmax())
confidence = float(prob[label_idx])
```

- 测试：`from fastapi.testclient import TestClient`，不用起服务。
