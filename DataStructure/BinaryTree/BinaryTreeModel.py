class BinaryTreeModel:
    def __init__(self, value, left = None, right = None):
        self.__value = value
        self.__left = left
        self.__right = right


    # getter方法（必须定义在前！！！）
    @property
    def value(self):
        return self.__value

    @property
    def left(self):
        return self.__left

    @property
    def right(self):
        return self.__right


    # setter方法（setter需定义在getter之后！！！）
    @value.setter
    def value(self, value):
        self.__value = value

    @left.setter
    def left(self, value):
        self.__left = value

    @right.setter
    def left(self, value):
        self.__left = value


    # 前序遍历方法（用于验证）
    def preorder_traversal(self):
        """
        根节点 =>  左子树 => 右子树
        :return:
        """

        """返回前序遍历结果"""
        result = [self.__value]
        if self.__left:
            result.extend(self.__left.preorder_traversal())
        if self.__right:
            result.extend(self.__right.preorder_traversal())
        return result

    def __str__(self):
        return f"BinaryTreeModel(value={self.__value}, left={self.__left}, right={self.__right})"


def init_data(data_list=[2, 3, 5, 7, 11, 13, 17, 19]):
    """
    将数据列表初始化为前序遍历格式的二叉树
    使用递归方式构建：第一个元素为根节点，然后递归构建左子树和右子树
    """
    if not data_list:
        return None

    # 取第一个元素作为根节点
    root_value = data_list[0]

    # 如果列表只有一个元素，直接返回叶子节点
    if len(data_list) == 1:
        return BinaryTreeModel(root_value)

    # 将剩余元素分为左右两部分
    # 这里采用简单的方式：左半部分和右半部分（可以调整策略）
    mid = len(data_list) // 2
    left_data = data_list[1:mid + 1]  # 左子树数据
    right_data = data_list[mid + 1:]  # 右子树数据

    # 递归构建左右子树
    left_tree = init_data(left_data) if left_data else None
    right_tree = init_data(right_data) if right_data else None

    return BinaryTreeModel(root_value, left_tree, right_tree)


# 测试代码
if __name__ == "__main__":
    data = [2, 3, 5, 7, 11, 13, 17, 19]

    # 初始化二叉树
    tree = init_data(data)

    # 验证前序遍历
    print("前序遍历结果:", tree.preorder_traversal())
    print("原始数据:", data)




