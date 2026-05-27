from datetime import date, datetime


# 当前完整时间（年-月-日 时:分:秒.毫秒）
now = datetime.now()
print(now)                                  # 2026-05-26 16:20:38.290880

# 指定时间格式（年-月-日 时:分:秒）
print(now.strftime('%Y-%m-%d %H:%M:%S'))    # 2026-05-26 16:25:34

# 指定时间格式（时:分:秒）
print(now.strftime('%H:%M:%S'))             # 16:27:36


# 当前日期（年月日）
today = date.today()
print(today)                                # 2026-05-26

# 指定时间格式（年月日）
print(now.strftime('%Y年%#m月%#d日'))        # 2026年5月27日
# 指定时间格式（年月日）
print(f"{now.year}年{int(now.strftime('%m'))}月{int(now.strftime('%d'))}日")       # 2026年5月27日
