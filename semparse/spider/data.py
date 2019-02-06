import json
import re
import qelos as q


def load_tables(p="../../data/spider/tables.json"):
    dbs = json.load(open(p))
    print(len(dbs))
    for db in dbs:
        print(db["db_id"])
        tables = {}
        fkeys = dict(db["foreign_keys"])
        i = 0
        multifkey = False
        for column in db["column_names"]:
            if column[0] not in tables:
                tables[column[0]] = set()
            tables[column[0]].add(i)
            i += 1
        for table_k, table in tables.items():
            # foreign keys for table
            table_fkeys = list(filter(lambda x: x in table, fkeys.keys()))
            table_fkey_tables = []  # tables reachable from this table using fkeys
            for table_fkey in table_fkeys:
                for table_j, table_other in tables.items():
                    if fkeys[table_fkey] in table_other:
                        table_fkey_tables.append(table_j)
            if len(table_fkey_tables) != len(set(table_fkey_tables)):
                multifkey = True

        print("{} foreign key mappings".format(len(fkeys)))
        if multifkey:
            print("multi-fkey connections")


def check_gold_sql(p="../../data/spider/train_gold.sql"):
    lines = open(p).readlines()
    lines = set(lines)
    # how many unique lines have T3?
    T2lines = []
    T3lines = []
    T4lines = []
    T5lines = []
    groupbylines = []
    nestedlines = []
    joinlines = []
    rejoinlines = []    # lines where different aliases are used for same table
    joinwithouttlines = []
    argmaxlines = []
    nonargmaxlines = []
    joinwithoutgroupby = []
    nonstarcountlines = []
    countlines = []
    countdistinctlines = []
    for line in lines:
        line = line.lower()
        if "t2." in line:
            assert("t1." in line)
            T2lines.append(line)
        if "t3." in line:
            assert("t2." in line)
            T3lines.append(line)
        if "t4." in line:
            assert("t3." in line)
            T4lines.append(line)
        if "t5." in line:
            assert("t4." in line)
            T5lines.append(line)
        if "group by" in line:
            groupbylines.append(line)
        if re.match(".+\(\s?select.+\).+", line):
            nestedlines.append(line)
        if "join" in line:
            joinlines.append(line)
            # get all table aliases
            selectsplits = line.split("select")
            rejoined = False
            for selectsplit in selectsplits:
                ms = re.findall("([^\s]+)\sas\s(t\d)", selectsplit)
                if len(list(zip(*ms))) > 0:
                    sms = set(list(zip(*ms))[0])
                    # print(len(ms) != len(sms))
                    if len(ms) > 0 and len(ms) != len(sms):
                        rejoined = True
                        # rejoinlines.append(line)
            if rejoined:
                rejoinlines.append(line)
            if not re.match(".+t\d\..+", line):
                joinwithouttlines.append(line)
            if not "group by" in line:
                joinwithoutgroupby.append(line)
            # print(len(ms))
        if re.match(".+limit\s\d+.+", line):
            if "limit 1" in line:
                argmaxlines.append(line)
            else:
                nonargmaxlines.append(line)
        if "count(" in line:
            countlines.append(line)
            if "count(*)" in line:
                pass
            if re.match(".+count\s?\(\s?distinct.+", line):
                countdistinctlines.append(line)
            if not "count(*)" in line and not re.match(".+count\s?\(\s?distinct.+", line):
                nonstarcountlines.append(line)

    print("{} unique sqls".format(len(lines)))
    print("{} unique sqls have T2".format(len(T2lines)))
    print("{} unique sqls have T3".format(len(T3lines)))
    print("{} unique sqls have T4".format(len(T4lines)))
    print("{} unique sqls have T5".format(len(T5lines)))
    print("{} unique sqls have GROUP BY".format(len(groupbylines)))
    print("{} unique sqls have nesting".format(len(nestedlines)))
    print("{} unique sqls have JOIN".format(len(joinlines)))
    print("{} unique sqls have JOIN without aliases".format(len(joinwithouttlines)))
    print("{} unique sqls have JOIN without GROUP BY".format(len(joinwithoutgroupby)))
    print("{} unique sqls have multiple aliases for same table".format(len(rejoinlines)))
    print("{} unique sqls have argmax".format(len(argmaxlines)))
    print("{} unique sqls have order by without argmax".format(len(nonargmaxlines)))
    print("{} unique sqls have COUNT(".format(len(countlines)))
    print("{} unique sqls have COUNT(DISTINCT ...)".format(len(countdistinctlines)))
    print("{} unique sqls with COUNT have no COUNT(*) or COUNT(DISTINCT(...))".format(len(nonstarcountlines)))

    # for line in joinwithoutgroupby:
    #     print(line)


if __name__ == '__main__':
    q.argprun(load_tables)
    # q.argprun(check_gold_sql)