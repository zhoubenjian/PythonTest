import math


def entropy(p_list):
    """
    计算熵 H = -sum(p_i * log2(p_i))
    p_list: 各个类别的概率列表，如 [0.5, 0.5]
    :return:
    """
    ent = 0.0
    for p in p_list:
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def gini(p_list):
    """
    计算基尼指数 Gini = 1 - sum(p_i^2)
    p_list: 各个类别的概率列表
    :return:
    """
    g = 1.0
    for p in p_list:
        if p > 0:
            g -= p * p
    return g


def info_gain(parent_p, child_p_list, child_weights):
    """
    计算信息增益
    parent_p：父节点概率分布
    child_p_list：子节点概率分布列表
    child_weights：每个子节点样本占父节点权重
    Gain = H(parent) - sum( weight * H(child) )
    """
    h_parent = entropy(parent_p)
    h_children = 0.0
    for p_child, w in zip(child_p_list, child_weights):
        h_children += w * entropy(p_child)
    return h_parent - h_children


def gini_gain(parent_p, child_p_list, child_weights):
    """基尼增益：分裂前后基尼的减少量"""
    g_parent = gini(parent_p)
    g_children = 0.0
    for p_child, w in zip(child_p_list, child_weights):
        g_children += w * gini(p_child)
    return g_parent - g_children



if __name__ == "__main__":
    # ========= 例子：父节点：正4，负4，共8个样本 =========
    parent_total = 8
    pos_p = 4 / parent_total
    neg_p = 4 / parent_total
    parent_prob = [pos_p, neg_p]

    print("====父节点（4正4负）====")
    print(f"熵 entropy = {entropy(parent_prob):.4f}")
    print(f"基尼 gini = {gini(parent_prob):.4f}\n")

    # 分裂之后：左子集全正(4正0负)；右子集全负(0正4负)
    # 两个子集各4个样本，权重都是4/8=0.5
    child1_prob = [1.0, 0.0]
    child2_prob = [0.0, 1.0]
    children_probs = [child1_prob, child2_prob]
    weights = [0.5, 0.5]

    ig = info_gain(parent_prob, children_probs, weights)
    gg = gini_gain(parent_prob, children_probs, weights)

    print("====分裂后：左全正，右全负====")
    print(f"信息增益 info_gain = {ig:.4f}")
    print(f"基尼增益 gini_gain = {gg:.4f}\n")

    # =========再模拟一个很差的分裂：分裂后两边依旧2正2负，没有提纯=========
    child_a = [0.5, 0.5]
    child_b = [0.5, 0.5]
    ig2 = info_gain(parent_prob, [child_a, child_b], [0.5,0.5])
    gg2 = gini_gain(parent_prob, [child_a, child_b], [0.5,0.5])
    print("====很差分裂：两边还是2正2负，纯度没提升====")
    print(f"信息增益 = {ig2:.4f}")
    print(f"基尼增益 = {gg2:.4f}")