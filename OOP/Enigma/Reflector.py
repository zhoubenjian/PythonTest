class Reflector:
    """反射器类（固定不旋转，实现加密可逆性）"""

    def __init__(self, wiring: str):
        """
        初始化反射器
        :param wiring: 反射器布线表（26个大写英文字母，无重复，对称映射）
        """
        if len(wiring) != 26 or not wiring.isalpha():
            raise ValueError("反射器布线必须是26个英文字母")

        self.wiring = wiring.upper()
        self.base_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def map(self, char: str) -> str:
        """反射器映射（接收转子输出，反向传回转子组）"""
        if not char.upper() in self.base_alphabet:
            return char

        char_index = self.base_alphabet.index(char.upper())
        return self.wiring[char_index]

    # 预设经典反射器（方便直接使用）
    @classmethod
    def reflector_b(cls):
        """经典 Enigma 反射器 B（最常用）"""
        return cls("YRUHQSLDPXNGOKMIEBFZCWVJAT")

    @classmethod
    def reflector_c(cls):
        """经典 Enigma 反射器 C"""
        return cls("FVPJIAOYEDRZXWGCTKUQSBNMHL")