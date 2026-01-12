# 测试示例
from OOP.Enigma.EnigmaMachine import EnigmaMachine
from OOP.Enigma.Reflector import Reflector
from OOP.Enigma.Rotor import Rotor


if __name__ == "__main__":
    # 步骤1：创建3个经典转子（Enigma I 型布线，初始位置均为A）
    rotor1 = Rotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "R", "A")
    rotor2 = Rotor("AJDKSIRUXBLHWTMCQGZNPYFVOE", "F", "A")
    rotor3 = Rotor("BDFHJLCPRTXVZNYEIWGAKMUSQO", "W", "A")
    rotors = [rotor1, rotor2, rotor3]

    # 步骤2：创建反射器（使用经典反射器B）
    reflector = Reflector.reflector_b()

    # 步骤3：初始化Enigma加密机（插线板配置：A↔B、C↔D、E↔F）
    enigma = EnigmaMachine(rotors, reflector, "AB CD EF")

    # 步骤4：测试加密/解密
    plain_text = "HELLO ENIGMA! THIS IS A TEST MESSAGE."
    print(f"明文：{plain_text}")

    # 加密
    cipher_text = enigma.process_text(plain_text)
    print(f"密文：{cipher_text}")

    # 步骤5：重置Enigma配置（解密需使用与加密完全相同的配置）
    rotor1_dec = Rotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "R", "A")
    rotor2_dec = Rotor("AJDKSIRUXBLHWTMCQGZNPYFVOE", "F", "A")
    rotor3_dec = Rotor("BDFHJLCPRTXVZNYEIWGAKMUSQO", "W", "A")
    enigma_dec = EnigmaMachine(
        [rotor1_dec, rotor2_dec, rotor3_dec],
        Reflector.reflector_b(),
        "AB CD EF"
    )

    # 解密
    decrypted_text = enigma_dec.process_text(cipher_text)
    print(f"解密后：{decrypted_text}")