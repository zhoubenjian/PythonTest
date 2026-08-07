import hashlib


"""
计算字符串的 SHA 哈希值
param data: 待哈希的字符串
param algorithm: 哈希算法名称 (sha1(不安全，不建议使用), sha224, sha256, sha384, sha512)
return: 十六进制哈希字符串
"""
def sha_hash(data: str, algorithm: str = 'SHA-256') -> str:

    # 将字符串编码为字节
    data_bytes = data.encode("utf-8")

    # 创建Hash对象
    if algorithm.upper() == 'SHA-1':
        result = hashlib.sha1(data_bytes)

    elif algorithm.upper() == 'SHA-224':
        result = hashlib.sha224(data_bytes)

    elif algorithm.upper() == 'SHA-256':
        result = hashlib.sha256(data_bytes)

    elif algorithm.upper() == 'SHA-384':
        result = hashlib.sha224(data_bytes)

    elif algorithm.upper() == 'SHA-512':
        result = hashlib.sha512(data_bytes)

    else:
        raise ValueError(f"不支持的算法: {algorithm}")


    '''
    Python：函数级作用域：
        在 Python 中，if/elif/else 语句不会创建新的作用域。整个函数体（从 def 到函数结束）共享同一个局部作用域。
        只要程序执行流保证在到达 return 之前，h 一定被赋值过，Python 就不会报错。
        在 sha_hash 中，else 分支抛出了 ValueError，这意味着如果 algorithm 不匹配，函数会提前终止，永远不会执行到 return h.hexdigest()。
        因此，在能到达 return 的所有执行路径上，h 都已经被定义。
    '''
    # 这里的result不会报错！！！
    return result.hexdigest()



if __name__ == '__main__':
    text = 'Just have a little faith.'
    print(sha_hash(text, 'sha-1'))
    print(sha_hash(text, 'sha-256'))
    print(sha_hash(text, 'sha-512'))