from OOP.Enigma.Reflector import Reflector
from OOP.Enigma.Rotor import Rotor


class EnigmaMachine:
    """Enigma 加密机核心类（整合所有组件，实现完整加密/解密流程）"""

    def __init__(self, rotors: list[Rotor], reflector: Reflector, plugboard_pairs: str = ""):
        """
        初始化 Enigma 加密机
        :param rotors: 转子组（列表形式，按加密流程顺序排列）
        :param reflector: 反射器实例
        :param plugboard_pairs: 插线板字符对（如 "AB CD EF"，表示A↔B、C↔D）
        """
        if not rotors or not all(isinstance(r, Rotor) for r in rotors):
            raise ValueError("转子组不能为空，且必须全部是Rotor实例")
        if not isinstance(reflector, Reflector):
            raise ValueError("必须传入有效的Reflector实例")

        self.rotors = rotors
        self.reflector = reflector
        self.base_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # 初始化插线板
        self.plugboard = self._initialize_plugboard(plugboard_pairs)

    def _initialize_plugboard(self, plugboard_pairs: str) -> dict:
        """初始化插线板（双向字符映射）"""
        # 初始化为字符自身映射
        plugboard = {char: char for char in self.base_alphabet}

        if not plugboard_pairs:
            return plugboard

        # 处理自定义插线对
        pairs = plugboard_pairs.strip().split()
        for pair in pairs:
            if len(pair) != 2 or not pair.isalpha():
                raise ValueError("插线板字符对必须是2个英文字母（如 AB）")

            char1 = pair[0].upper()
            char2 = pair[1].upper()
            if char1 == char2:
                continue  # 跳过自身映射

            # 双向绑定（A↔B 等同于 B↔A）
            plugboard[char1] = char2
            plugboard[char2] = char1

        return plugboard

    def _rotate_rotors(self):
        """转子旋转进位（模拟机械齿轮进位逻辑）"""
        # 从第一个转子开始，依次判断是否需要进位
        for rotor in self.rotors:
            is_carry = rotor.rotate()
            if not is_carry:
                break  # 当前转子未触发进位，后续转子不旋转

    def process_char(self, char: str) -> str:
        """加密/解密单个字符（Enigma 核心流程）"""
        upper_char = char.upper()
        if upper_char not in self.base_alphabet:
            return char

        # 步骤1：插线板正向替换
        current_char = self.plugboard[upper_char]

        # 步骤2：转子组正向映射（从第一个到最后一个）
        for rotor in self.rotors:
            current_char = rotor.forward_map(current_char)

        # 步骤3：反射器映射
        current_char = self.reflector.map(current_char)

        # 步骤4：转子组反向映射（从最后一个到第一个）
        for rotor in reversed(self.rotors):
            current_char = rotor.backward_map(current_char)

        # 步骤5：插线板反向替换（利用双向映射特性）
        current_char = self.plugboard[current_char]

        # 步骤6：转子旋转进位
        self._rotate_rotors()

        return current_char

    def process_text(self, input_text: str) -> str:
        """加密/解密字符串（自动转换为大写处理）"""
        if not input_text:
            return ""

        output_chars = []
        for char in input_text:
            output_chars.append(self.process_char(char))

        return ''.join(output_chars)