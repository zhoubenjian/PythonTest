'''
转子（Enigma 核心动态加密组件）
'''


class Rotor:
    """转子类（Enigma 核心动态加密组件）"""

    def __init__(self, forward_wiring: str, notch_char: str, initial_position: str = 'A'):
        """
        初始化转子
        :param forward_wiring: 正向布线表（26个大写英文字母，无重复）
        :param notch_char: 进位字符（触发下一个转子旋转的标识）
        :param initial_position: 转子初始位置（默认'A'）
        """
        # 验证输入合法性
        if len(forward_wiring) != 26 or not forward_wiring.isalpha():
            raise ValueError("转子正向布线必须是26个英文字母")
        if not (notch_char.isalpha() and len(notch_char) == 1):
            raise ValueError("进位字符必须是单个英文字母")
        if not (initial_position.isalpha() and len(initial_position) == 1):
            raise ValueError("初始位置必须是单个英文字母")

        # 统一转换为大写，确保字符一致性
        self.forward_wiring = forward_wiring.upper()
        self.notch_char = notch_char.upper()
        self.base_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # 构建反向布线表（正向映射的逆过程）
        self.backward_wiring = [''] * 26
        for idx, char in enumerate(self.forward_wiring):
            char_index = self.base_alphabet.index(char)
            self.backward_wiring[char_index] = self.base_alphabet[idx]
        self.backward_wiring = ''.join(self.backward_wiring)

        # 初始化转子当前位置（转换为0-25的索引值）
        self.current_position = self.base_alphabet.index(initial_position.upper())
        # 初始化进位位置（转换为0-25的索引值）
        self.notch_position = self.base_alphabet.index(self.notch_char)

    def _offset_index(self, char: str) -> int:
        """辅助方法：将字符转换为考虑转子偏移的索引"""
        char_index = self.base_alphabet.index(char.upper())
        return (char_index + self.current_position) % 26

    def _restore_index(self, offset_index: int) -> str:
        """辅助方法：将偏移后的索引还原为对应字符"""
        original_index = (offset_index - self.current_position + 26) % 26
        return self.base_alphabet[original_index]

    def forward_map(self, char: str) -> str:
        """转子正向映射（明文→反射器方向）"""
        if not char.upper() in self.base_alphabet:
            return char

        # 计算偏移后的输入索引，获取正向映射结果，还原偏移并返回字符
        offset_input_idx = self._offset_index(char)
        mapped_char = self.forward_wiring[offset_input_idx]
        return self._restore_index(self.base_alphabet.index(mapped_char))

    def backward_map(self, char: str) -> str:
        """转子反向映射（反射器→明文方向）"""
        if not char.upper() in self.base_alphabet:
            return char

        # 计算偏移后的输入索引，获取反向映射结果，还原偏移并返回字符
        offset_input_idx = self._offset_index(char)
        mapped_char = self.backward_wiring[offset_input_idx]
        return self._restore_index(self.base_alphabet.index(mapped_char))

    def rotate(self) -> bool:
        """
        旋转转子，返回是否触发进位
        :return: True=触发下一个转子进位，False=不触发
        """
        # 旋转前判断是否处于进位位置（当前位置==进位位置，旋转后触发进位）
        is_carry = (self.current_position == self.notch_position)
        # 更新转子当前位置（循环偏移，0-25）
        self.current_position = (self.current_position + 1) % 26
        return is_carry