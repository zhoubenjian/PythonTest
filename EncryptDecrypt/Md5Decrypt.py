import hashlib


def md5_hash(data):
    # 将字符串编码为字节
    data_bytes = data.encode("utf-8")
    # 实现加密
    return hashlib.md5(data_bytes).hexdigest()



if __name__ == '__main__':
    text = 'Everything is goes well!'
    print(md5_hash(text))