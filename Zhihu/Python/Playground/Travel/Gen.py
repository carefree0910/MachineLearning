# encoding: utf-8
import pickle


# ======================================================================================================================
# Generate Information

def get_vec(vectors, names, labels, label_set):
    for i in names.items():
        label, info = i[1][1], i[1][2]
        dic = {}
        s = 0
        l_sub = []
        for j in range(len(label_set)):
            t_l = labels[j + 1]
            if t_l in label:
                l_sub.append(j)
            c = info.count(t_l)
            if c > 0:
                dic[t_l] = (j, c)   # j is subscript, c is weight
                s += c
        s *= 2
        vec = vectors[i[0]]
        for j in dic.values():
            sub, weight = j
            vec[sub] = weight / s
        s = 0
        for j in vectors[i[0]].values():
            s += j
        big = 0.5 / len(l_sub) if s != 0 else 1 / len(l_sub)
        for j in l_sub:
            if j in vec:
                vec[j] += big
            else:
                vec[j] = big
    return vectors


def get_info():
    try:
        with open("Data/Info.dat", "rb") as file:
            names, labels, label_set, locs, loc_set = pickle.load(file)
        file.close()
    except FileNotFoundError:
        data = open("Data/Spots.txt", "r", encoding="utf-8").readlines()
        if not data:
            data = open("Data/Storage/Spots.txt", "r", encoding="utf-8").readlines()
        all_loc = {"东城区", "西城区", "朝阳区", "丰台区", "海淀区", "亦庄开发区", "燕山石化区", "门头沟区",
                   "通州区", "平谷区", "石景山区", "怀柔区", "延庆县", "顺义区", "大兴区", "昌平区", "密云县", "房山区"}
        names, loc_set, label_set = dict(), set(), set()
        for i in range(len(data) // 7):
            t_grid, t_loc, t_label, t_info, t_trans, t_vehicle = \
                data[7 * i + 1].split(","), data[7 * i + 2].strip(), \
                data[7 * i + 3].strip().split("="), data[7 * i + 4].strip(), data[7 * i + 5].strip(), \
                data[7 * i + 6]
            for j in all_loc:
                if j in t_loc:
                    loc_set.add(j)
                    break
            for j in t_label:
                label_set.add(j.strip())
            names[data[7 * i].strip()] = [t_loc, t_label, t_info, t_trans, t_grid, t_vehicle]

        labels = [i for i in label_set]
        labels.sort()
        labels.insert(0, "Type")
        locs = [i for i in loc_set]
        locs.sort()
        locs.insert(0, "Location")

        with open("Data/Info.dat", "wb") as file:
            pickle.dump((names, labels, label_set, locs, loc_set), file)
        file.close()

    try:
        with open("Data/Vector.dat", "rb") as file:
            vectors, total_vector = pickle.load(file)
        file.close()
    except FileNotFoundError:
        total_vector = [0] * len(label_set)
        vectors = {i: {} for i in names}
        vectors = get_vec(vectors, names, labels, label_set)

        with open("Data/Vector.dat", "wb") as file:
            pickle.dump((vectors, total_vector), file)
        file.close()

    return names, labels, label_set, locs, loc_set, vectors, total_vector


# ======================================================================================================================
# Generate Roads

def get_distance(delta_w, delta_j):
    return abs(111 * delta_w) + abs(85.2 * delta_j) + 0.00001


def jwd_distance(jwd1, jwd2):
    return get_distance(jwd1[0] - jwd2[0], jwd1[1] - jwd2[1])


def direction(jwd1, jwd2):
    dy = 111 * (jwd2[0] - jwd1[0])
    dx = 85.2 * (jwd2[1] - jwd1[1])
    if dx / (-2) < dy < dx / 2:
        return '向东'
    if dx / 2 < dy < dx / (-2):
        return '向西'
    if dy / (-2) < dx < dy / 2:
        return '向北'
    if dy / 2 < dx < dy / (-2):
        return '向南'
    if dx > 0 and dy > 0:
        return '向东北方向'
    if dx > 0 and dy < 0:
        return '向东南方向'
    if dx < 0 and dy > 0:
        return '向西北方向'
    if dx < 0 and dy < 0:
        return '向西南方向'
    return ''


def get_sub_data():
    # ['科怡路', ['9号线'], [39.9261720655, 116.1905462972], {'687', '694', '604', '740', '340'}]
    ans = []
    sub_txt = open('Data/Subway.txt', 'r', encoding='utf-8')
    lines = sub_txt.readlines()
    sub_txt.close()
    for i in range(329):
        name = lines[4*i][:-1]
        if i == 0:
            name = lines[4*i][1:-1]
        num = lines[4*i+1][:-1].split()
        jwd = [float(lines[4*i+2][:-1].split()[j]) for j in range(2)]
        buses = set(lines[4*i+3][:-1].split())
        ans.append([name, num, jwd, buses])
    return ans


def get_bus_data():
    ans = {}
    bus_txt = open('Data/Bus.txt', 'r', encoding='utf-8')
    lines = bus_txt.readlines()
    bus_txt.close()
    for bus_i in range(int(len(lines)/2)):
        lu = lines[2*bus_i][:-1].split()[0]
        sta = lines[2*bus_i][:-1].split()[1]
        jwd_str = lines[2*bus_i+1][:-1].split(',')
        jwd = [float(jwd_str[1])-0.0058, float(jwd_str[0])-0.0065]
        if lu in ans:
            ans[lu].append([sta, lu, jwd])
        else:
            ans[lu] = [[sta, lu, jwd]]
    return ans


def get_price(dis, tool):  # 2015 北京地铁、公交计价标准
    if tool == 'bus':
        if dis < 10:
            return 2
        elif dis > 40:
            return 9
        else:
            return int(dis / 5) + 1
    elif tool == 'subway':
        if dis < 6:
            return 3
        elif dis < 12:
            return 4
        else:
            return int((dis - 2) / 10) + 4
    elif tool == 'walk':
        return 0
    else:
        print('tool is "subway" or "bus" or "walk"')


def get_time(dis, tool):  # min
    if tool == 'bus':
        return int(dis / (31.3 / 60))
    elif tool == 'subway':
        return int(dis / (41.5 / 60))
    elif tool == 'walk':  # <1km
        return int(dis / (4.3 / 60))
    else:
        print('tool is "subway" or "bus" or "walk"')


def bus_subs(data):
    ans = [set() for i in range(1000)]
    for station in data:
        for bus in station[3]:
            ans[int(bus)].add((station[0], tuple(station[1]), tuple(station[2]), tuple(station[3])))
    return ans


def subway_may(sta_name1, sta_name2, data):
    subways = ['1号线', '2号线', '4号线', '5号线', '6号线', '7号线', '8号线', '9号线', '10号线', '13号线',
           '14号线', '15号线', '八通线', '昌平线', '亦庄线', '房山线', '机场线', '大兴线']
    num_dic = {subways[i]: i for i in range(18)}
    sub_max = [[0 for i in range(18)]for j in range(18)]
    tran_max = [[set() for i in range(18)] for j in range(18)]
    for station in data:
        if station[0] == sta_name1:
            num_list1 = station[1]
        if station[0] == sta_name2:
            num_list2 = station[1]
        if len(station[1]) == 1:
            continue
        for num1 in station[1]:
            num1_i = num_dic[num1]
            for num2 in station[1]:
                num2_i = num_dic[num2]
                if num1_i != num2_i:
                    sub_max[num1_i][num2_i] = 1
                    tran_max[num1_i][num2_i].add(station[0])
    if set(num_list1) & set(num_list2) != set():
        for i in set(num_list1) & set(num_list2):
            return [[[i]], [[]]]
    visited = set()
    dalao_list = [[]]
    for num in num_list1:
        dalao_list[0].append([num_dic[num], set()])
        visited.add(num_dic[num])
    get = False
    while not get:
        olds = dalao_list[-1]
        dalao_list.append([])
        for old in olds:
            for i in range(18):
                if sub_max[old[0]][i] == 1: #and i not in visited:
                    i_appear = False
                    for j in range(len(dalao_list[-1])):
                        if dalao_list[-1][j][0] == i:
                            i_appear = str(j)
                    if i_appear:
                        i_appear = int(i_appear)
                        dalao_list[-1][i_appear][1].add(old[0])
                    else:
                        for k in num_list2:
                            if num_dic[k] == i:
                                get = True
                        visited.add(i)
                        dalao_list[-1].append([i, set([old[0]])])
    ans = []
    for a in num_list2:
        ans_a = [[num_dic[a]]]
        for b in range(len(dalao_list)-1, 0, -1):
            list_b = dalao_list[b]
            new_ans_a = []
            for way_list in ans_a:
                father = way_list[0]
                for small_list in list_b:
                    if father == small_list[0]:
                        for c in small_list[1]:
                            new_aaa = [c]
                            for d in way_list:
                                new_aaa.append(d)
                            new_ans_a.append(new_aaa)
            ans_a = new_ans_a
        for i in ans_a:
            ans.append(i)
    tran_stas = []
    def sub_distance(name1, name2, data):
        for _station in data:
            if _station[0] == name1:
                _jwd1 = _station[2]
            if _station[0] == name2:
                _jwd2 = _station[2]
        return jwd_distance(_jwd1, _jwd2)
    for i in range(len(ans)):
        tran_stas.append([])
        for j in range(len(ans[i])-1):
            may_tran = list(tran_max[ans[i][j]][ans[i][j+1]])
            tran_stas[i].append(may_tran[0])
            if len(may_tran) == 2:
                station_name1 = sta_name1
                station_name3 = sta_name2
                station_name2 = may_tran[0]
                dis_old = sub_distance(station_name1, station_name2, data) + sub_distance(station_name2, station_name3, data)
                station_name2 = may_tran[1]
                dis_new = sub_distance(station_name1, station_name2, data) + sub_distance(station_name2, station_name3, data)
                if dis_new < dis_old:
                    del tran_stas[i][-1]
                    tran_stas[i].append(may_tran[1])
    for i in range(len(ans)):
        for j in range(len(ans[i])):
            ans[i][j] = subways[ans[i][j]]
    return [ans, tran_stas]


def near_subway(jwd, data):
    best_dis = pri_dis = 1.5
    for station in data:
        sta_jwd = station[2]
        dis = get_distance(sta_jwd[0] - jwd[0], sta_jwd[1] - jwd[1])
        if dis < best_dis:
            best_station = station
            best_dis = dis
    if best_dis < pri_dis:
        return best_station
    else:
        return None


def good_stas(jwd, bus_list, bus_sub_list):
    stas = []
    visited = set()
    for bus in bus_list:
        for sta in bus_sub_list[bus]:
            if sta[0] not in visited:
                stas.append([sta, bus])
                visited.add(sta[0])
    if not stas:
        return None
    ans = []
    while len(ans) < 4 and stas:
        best = stas[0]
        best_jwd = best[0][2]
        for sta_and_bus in stas:
            new_jwd = sta_and_bus[0][2]
            if jwd_distance(new_jwd, jwd) < jwd_distance(best_jwd, jwd):
                best = sta_and_bus
                best_jwd = new_jwd
        if not ans or best != ans[-1]:
            ans.append(best)
            stas.remove(best)
            del_list = []
            for num in best[0][1]:
                for sta_and_bus in stas:
                    if len(sta_and_bus[0][1]) == 1 and str(num) in sta_and_bus[0][1]:
                        del_list.append(sta_and_bus)
            for i in range(len(stas)-1, -1, -1):
                if stas[i] in del_list:
                    del stas[i]
    return ans


def get_good(message1, message2, love, info_dic, data, bus_data):
    jwd1 = [float(info_dic[message1][4][i][:-1]) for i in range(2)]
    bus1 = []
    for i in info_dic[message1][5].split():
        try:
            i = int(i)
            bus1.append(i)
        except:
            pass
    jwd2 = [float(info_dic[message2][4][i][:-1]) for i in range(2)]
    bus2 = []
    for i in info_dic[message2][5].split():
        try:
            i = int(i)
            bus2.append(i)
        except:
            pass
    bus_sub_list = bus_subs(data)
    if jwd_distance(jwd1, jwd2) < 1:
        dis = int(jwd_distance(jwd1, jwd2)*100)*10
        dir = direction(jwd1, jwd2)
        if dis < 10:
            dis = 50
        return ['{}步行{}米'.format(dir, dis), get_time(dis/1000, 'walk'), 0]
    for bus in bus1:
        if bus in bus2:
            dis = jwd_distance(jwd1, jwd2)
            return ['乘坐公交{}路'.format(bus), 10+get_time(dis, 'bus'), get_price(dis, 'bus')]
    to_sub1 = near_subway(jwd1, data)
    # ['科怡路', ['9号线'], [39.9261720655, 116.1905462972], {'687', '694', '604', '740', '340'}]
    if to_sub1:
        to_sub1 = [[(to_sub1[0], tuple(to_sub1[1]), tuple(to_sub1[2]), tuple(to_sub1[3])), None]]
    else:
        to_sub1 = good_stas(jwd1, bus1, bus_sub_list)
    # to_sub1 may sta_and_bus list or [] # bus maybe 1,2 or None
    to_sub2 = near_subway(jwd2, data)
    if to_sub2:
        to_sub2 = [[(to_sub2[0], tuple(to_sub2[1]), tuple(to_sub2[2]), tuple(to_sub2[3])), None]]
    else:
        to_sub2 = good_stas(jwd2, bus2, bus_sub_list)
    if not to_sub1 or not to_sub2:
        return 'no'
    final_list = []
    for start in to_sub1:
        for finish in to_sub2:
            if start[0][0] == finish[0][0]:
                # 不用进地铁
                time_all = 0
                price_all = 0
                nosub_str = ''
                dis = int(jwd_distance(jwd1, start[0][2])*100)*10
                if start[1] is None:
                    nosub_str += '步行{}米至{}站, '.format(dis, start[0][0])
                    time_all += get_time(dis/1000, 'walk')
                else:
                    nosub_str += '乘坐公交{}路至{}下车, '.format(start[1], start[0][0])
                    price_all += get_price(dis/1000, 'bus')
                    time_all += get_time(dis/1000, 'bus')
                dis = int(jwd_distance(jwd2, finish[0][2])*100)*10
                if finish[1] is None:
                    nosub_str += '步行{}米'.format(dis)
                    time_all += get_time(dis/1000, 'walk')
                else:
                    price_all += get_price(dis/1000, 'bus')
                    time_all += get_time(dis/1000, 'bus')
                    if start[1] is None:
                        nosub_str += '乘坐公交{}路'.format(finish[1])
                    else:
                        nosub_str += '换乘公交{}路'.format(finish[1])
                        time_all += 10
                return [nosub_str, time_all, price_all]
            sta_name1 = start[0][0]
            sta_name2 = finish[0][0]
            betweens = subway_may(sta_name1, sta_name2, data)
            for sub_n in range(len(betweens[0])):
                between = [betweens[0][sub_n], betweens[1][sub_n]]
                time_all = 0
                price_all = 0
                str_all = ''
                dis = int(jwd_distance(jwd1, start[0][2])*100)*10
                if start[1] is None:
                    dir = direction(jwd1, start[0][2])
                    str_all += '{}步行{}米至{}站, '.format(dir, dis, start[0][0])
                    time_all += get_time(dis/1000, 'walk')
                else:
                    str_all += '乘坐公交{}路至{}, '.format(start[1], start[0][0])
                    price_all += get_price(dis/1000, 'bus')
                    time_all += get_time(dis/1000, 'bus')
                sub_trans_jwd = [start[0][2]]
                for sta_name in between[1]:
                    for station in data:
                        if sta_name == station[0]:
                            sub_trans_jwd.append(station[2])
                            break
                sub_trans_jwd.append(finish[0][2])
                dis = jwd_distance(start[0][2], finish[0][2])
                price_all += get_price(dis, 'subway')
                dis = 0
                assert len(sub_trans_jwd) == 1+len(between[0])
                for i in range(len(sub_trans_jwd)-1):
                    dis += jwd_distance(sub_trans_jwd[i], sub_trans_jwd[i+1])
                time_all += 10 * len(between[1])
                time_all += get_time(dis, 'subway')
                str_all += '乘坐地铁{}, '.format(between[0][0])
                for i in range(len(between[1])):
                    str_all += '坐到{}站, 换乘地铁{}, '.format(between[1][i], between[0][i+1])
                str_all += '坐到{}站下车, '.format(finish[0][0])
                dis = int(jwd_distance(jwd2, finish[0][2])*100)*10
                if finish[1] is None:
                    dir = direction(finish[0][2], jwd2)
                    str_all += '{}步行{}米'.format(dir, dis)
                    time_all += get_time(dis/1000, 'walk')
                else:
                    str_all += '换乘公交{}路'.format(finish[1])
                    price_all += get_price(dis/1000, 'bus')
                    time_all += get_time(dis/1000, 'bus')
                final_list.append([str_all, time_all, price_all])
    #### final_list.append([str_all, time_all, price_all])
    near_bus1 = set()
    near_bus2 = set()
    for bus in bus_data:
        for j in bus_data[bus]:
            if jwd_distance(j[2], jwd1) < 1 and bus not in near_bus1:
                near_bus1.add(bus)
            if jwd_distance(j[2], jwd2) < 1 and bus not in near_bus2:
                near_bus2.add(bus)
    if near_bus1 & near_bus2:
        for bus in near_bus1 & near_bus2:
            dis = int(jwd_distance(jwd1, jwd2)*100)*10
            return ['乘坐公交{}路'.format(bus), 10+get_time(dis, 'bus'), get_price(dis, 'bus')]
    getable_sta1_part = set()
    getable_sta1_all = set()
    getable_sta2_part = set()
    getable_sta2_all = set()
    for bus in near_bus1:
        stas = bus_data[bus]
        for i in stas:
            getable_sta1_part.add((i[0], tuple(i[2])))
            getable_sta1_all.add((i[0], i[1], tuple(i[2])))
    for bus in near_bus2:
        stas = bus_data[bus]
        for i in stas:
            getable_sta2_part.add((i[0], tuple(i[2])))
            getable_sta2_all.add((i[0], i[1], tuple(i[2])))
    for sta_both in getable_sta1_part & getable_sta2_part:
        for sta in getable_sta1_all:
            if sta[0] == sta_both[0]:
                both_bus1 = sta[1]
                break
        for sta in getable_sta2_all:
            if sta[0] == sta_both[0]:
                both_bus2 = sta[1]
                break
        if sta_both[0][-1] == '站':
            str_all = '乘坐公交{}至{}, 换乘公交{}'.format(both_bus1, sta_both[0], both_bus2)
        else:
            str_all = '乘坐公交{}至{}站, 换乘公交{}'.format(both_bus1, sta_both[0], both_bus2)
        both_jwd = sta_both[1]
        time_all = 20 + get_time(jwd_distance(both_jwd, jwd1), 'bus') + get_time(jwd_distance(both_jwd, jwd2), 'bus')
        price_all = get_price(jwd_distance(both_jwd, jwd1), 'bus') + get_price(jwd_distance(both_jwd, jwd2), 'bus')
        final_list.append([str_all, time_all, price_all])

    best_time_i = 0
    best_price_i = 0
    best_tran_i = 0
    best_price = 10000
    best_time = 10000
    best_tran = 10000
    for i in range(len(final_list)):
        way = final_list[i]
        if way[0].count(',') < best_tran:
            best_tran = way[0].count(',')
            best_tran_i = i
        if way[1] < best_time:
            best_time = way[1]
            best_time_i = i
        if way[2] < best_price:
            best_price = way[2]
            best_price_i = i
    if love == 'change':
        return final_list[best_tran_i]
    elif love == 'time':
        return final_list[best_time_i]
    elif love == 'cost':
        return final_list[best_price_i]


def get_ways(mess_list, love, info_dic):
    data = get_sub_data()
    bus_data = get_bus_data()
    if len(mess_list) == 1:
        print('mess_list len wrong')
    ans_names = [mess_list[0]]
    del mess_list[0]
    ans_lists = []
    while mess_list:
        best = 1000
        best_i = 0
        a_point = ans_names[-1]
        goods = []
        for ii in range(len(mess_list)):
            another_point = mess_list[ii]
            good = get_good(a_point, another_point, love, info_dic, data, bus_data)
            if good == 'no':
                return 'They are unreachable, may because the new plot have imperfect information.'
            goods.append(good)
        assert len(mess_list) == len(goods)
        for ii in range(len(goods)):
            good = goods[ii]
            if love == 'change' and good[0].count(',') < best:
                best = good[0].count(',')
                best_i = ii
            if love == 'time' and good[1] < best:
                best = good[1]
                best_i = ii
            if love == 'cost' and good[2] < best:
                best = good[2]
                best_i = ii

        ans_lists.append(goods[best_i])
        ans_names.append(mess_list[best_i])
        del mess_list[best_i]
    ans_strs = '从{}出发, '.format(ans_names[0])
    for ii in range(len(ans_lists)):
        if ii != 0:
            if ii != len(ans_lists)-1 or ii == 1:
                ans_strs += '然后, '
            else:
                ans_strs += '最后, '
        ans_strs += ans_lists[ii][0]
        ans_strs += '至{}\n'.format(ans_names[ii+1])
    if len(ans_names) == 2:
        ans_strs += '预计时间：{}分钟\n'.format(ans_lists[0][1])
        ans_strs += '预计花费：{}元\n'.format(ans_lists[0][2])
    return ans_strs
