'''
    Name: DataBase Tools
    Creation: 2020-02-22
'''

import psycopg2 as pg

# 数据库的连接
# 返回conn和cur
def connect():
    conn = pg.connect(database="postgres",
                      user="postgres",
                      password="postgres",
                      host="localhost",
                      port="5432")

    cur = conn.cursor()

    return conn, cur


def executeSQL(sql):
    try:
        conn, cur = connect()
        cur.execute(sql)
    except Exception as e:
        print(e)
        return -1
    finally:
        cur.close()
        conn.close()

    return 0

def querySQL(sql):
    try:
        conn, cur = connect()
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        print(e)
        return None
    finally:
        cur.close()
        conn.close()

    return rows


