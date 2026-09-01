#!/usr/bin/env python3
"""Build deterministic, semantically isolated SANA LoRA prompt catalogs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import (  # noqa: E402
    PROMPT_SCHEMA_VERSION,
    validate_prompt_catalog,
    write_jsonl_atomic,
)


SUBJECT_CATALOG = {
    "kitchen_specialty": [
        "铜制摩卡壶", "手压柑橘榨汁器", "陶瓷黄油盒", "竹柄章鱼烧盘", "铸铁玉子烧锅", "双耳奶锅",
        "木质面包切片架", "旋柄苹果削皮机", "珐琅香料研磨钵", "玻璃蜂蜜分配器", "折叠蒸笼架", "手摇面条压制机",
        "不锈钢饺子模具", "陶瓷发酵罐", "木柄华夫饼夹", "台式冰淇淋压勺",
    ],
    "furniture_specialty": [
        "弧面唱片收纳柜", "三角转角书架", "翻盖针线收纳凳", "矮脚藤面换鞋凳", "双层盆景展示台", "折叠棋盘桌",
        "波浪门床尾柜", "圆柱抽屉边桌", "梯形杂志架", "拱门儿童衣柜", "旋转领带收纳架", "壁挂折叠餐桌",
        "马鞍形脚凳", "双杆被毯架", "半月玄关桌", "模块化六边形边柜",
    ],
    "lighting_specialty": [
        "蘑菇罩夜灯", "双环摄影补光灯", "折纸鹤壁灯", "矿工帽夹灯", "磁吸轨道射灯", "手提风暴油灯",
        "水母形吊灯", "三脚测绘探照灯", "书页夹阅读灯", "竹编球形顶灯", "熔岩玻璃氛围灯", "伸缩露营串灯卷轴",
        "棱镜床头灯", "月相投影夜灯", "陶瓷房屋香薰灯", "船舱防爆壁灯",
    ],
    "precision_tools": [
        "棘轮扭矩扳手", "手动铆钉枪", "弓形线锯", "木工燕尾标记规", "皮革旋转打孔钳", "玻璃吸盘搬运器",
        "钟表开盖器", "台式砂带打磨机", "管道扩口器", "折叠六角扳手组", "手摇台钻模型", "瓷砖切割钳",
        "双轮滚花刀", "木柄雕刻木槌", "弹簧测力计", "精密划线高度规",
    ],
    "consumer_devices": [
        "电子墨水记事板", "折叠旅行路由器", "胶片负片扫描仪", "桌面标签打印机", "手持热成像仪", "磁带随身听",
        "便携照片打印机", "机械翻页计时器", "指针式气象台", "掌上短波接收机", "双镜头行车记录仪", "桌面电子显微镜",
        "电容笔充电座", "袖珍录音转写器", "智能戒指充电盒", "复古圆屏示波器",
    ],
    "mobility_devices": [
        "三轮货运自行车", "单轮平衡车", "雪地履带摩托", "折叠皮划艇拖车", "机场行李牵引车", "站立式高尔夫球车",
        "电动独轮手推车", "沙滩风帆车", "窄轨矿山车", "履带式楼梯搬运车", "四轮儿童脚踏车", "水陆两用遥控车",
        "手摇轨道巡检车", "复古边斗摩托车", "双轮仓库拣货车", "折叠雪橇推车",
    ],
    "mechanical_toys": [
        "铁皮啄木鸟玩具", "发条跳蛙玩具", "木质齿轮钟玩具", "磁悬浮陀螺玩具", "手摇木马音乐盒", "机械翻跟斗猴玩具",
        "橡皮筋动力船模", "连杆步行蟹玩具", "太阳能甲虫玩具", "弹簧拳击袋鼠玩具", "齿轮开花盒玩具", "风力行走兽模型",
        "牵线木偶骑士", "滚珠迷宫塔", "平衡鸟科学玩具", "折叠纸板潜艇模型",
    ],
    "decor_objects": [
        "珐琅旋转音乐地球", "木质潮汐时钟", "玻璃气压风暴瓶", "黄铜日晷摆件", "陶瓷月兔香插", "石雕漩涡书挡",
        "铜制万花筒", "木框沙画摆件", "玻璃克莱因瓶模型", "铸铁渡鸦门挡", "陶瓷蘑菇储蓄罐", "黄铜机械太阳系仪",
        "木质 perpetual 日历", "彩玻璃捕光器", "石膏建筑柱头模型", "金属悬浮平衡雕塑",
    ],
    "special_containers": [
        "分格种子收纳箱", "旋盖胶片罐", "皮革望远镜筒", "木质雪茄保湿盒", "金属邮筒模型", "陶瓷橄榄油壶",
        "玻璃标本保存罐", "竹编茶饼提篮", "黄铜药粉盒", "帆布工具卷袋", "双层首饰旅行箱", "铝制唱片运输箱",
        "木桶形雨伞架", "折叠画笔清洗桶", "带锁文件手提箱", "六角蜂巢储物盒",
    ],
    "musical_instruments": [
        "拇指琴", "手碟鼓", "爱尔兰哨笛", "蛇形低音号", "轮擦提琴", "陶笛",
        "班卓尤克里里", "框架式竖琴", "木鱼套组", "巴西雨声筒", "手摇风琴", "钢舌鼓",
        "非洲卡林巴箱琴", "玻璃水晶琴", "蒙古马头琴", "机械音乐盒滚筒",
    ],
    "sports_equipment": [
        "冰壶石", "击剑护面", "反曲弓瞄准器", "花式滑水板", "桌上冰球球门", "攀岩上升器",
        "速度滑冰冰刀", "马术障碍杆座", "皮划艇脚踏舵", "飞盘高尔夫篮", "壁球球拍", "雪崩探测杆",
        "保龄球回球架模型", "竞技弹弓", "自由潜水脚蹼", "体操鞍马模型",
    ],
    "garden_devices": [
        "脚踏式堆肥翻转器", "铸铁软管导向轮", "手摇种子播种器", "玻璃自动浇水球", "盆栽土壤筛", "嫁接刀收纳架",
        "雨量计支架", "蜂箱烟熏器", "园艺球根种植器", "壁挂鸟食台", "旋转草坪喷头", "手推落叶收集滚筒",
        "陶瓷青蛙蓄水器", "折叠苗床压土器", "温室开窗器", "树木测径钳",
    ],
    "office_objects": [
        "滚轮日期印章", "铸铁票据压针座", "木质索引卡抽屉", "旋转名片展示架", "桌面封蜡炉", "手动凸字标签机",
        "折叠文件打孔器", "黄铜信件开封刀座", "机械号码印章", "多层橡皮章转盘", "桌面图纸卷筒架", "木质邮票湿润器",
        "夹式阅读稿架", "复古支票写字机", "旋柄卷笔刀", "金属纸张打孔规",
    ],
    "bath_objects": [
        "壁挂剃须刷架", "脚踏挤牙膏器", "陶瓷浴盐研磨罐", "折叠浴缸书架", "手摇毛巾拧干器", "黄铜浴缸塞链架",
        "双杯漱口架", "木质肥皂切割器", "旋转棉签分配盒", "珐琅便携洗脸盆", "壁挂浴帽收纳篮", "竹制足浴桶盖",
        "复古手压花洒泵", "瓷质剃须皂碗", "折叠旅行镜盒", "机械浴室秤",
    ],
    "outdoor_gear": [
        "折叠火焰反射板", "手摇帐篷打桩器", "铝制雪地锚", "皮革地图筒", "求生线锯卷盒", "便携风向袋架",
        "登山冰镐保护套", "折叠营地洗手台", "太阳能净水蒸馏器模型", "吊床树带卷轴", "铸铁荷兰锅支架", "露营咖啡滤架",
        "野外昆虫观察盒", "便携鱼线绕线器", "木质营地调料箱", "折叠雪鞋",
    ],
    "science_models": [
        "牛顿摆阵列", "手摇静电起电机", "太阳高度测量仪", "机械波演示器", "双盘偏振演示器", "地震记录仪模型",
        "阿基米德螺旋模型", "液体连通器教具", "陀螺仪万向架", "电磁钟摆实验器", "双锥体上滚演示器", "傅科摆桌面模型",
        "蒸汽机剖面模型", "晶体结构拼装架", "光谱转盘", "风洞烟线演示器",
    ],
    "industrial_components": [
        "蜗轮减速箱剖模", "法兰式蝶阀", "链条张紧器", "气动三联件", "手轮闸阀", "凸轮分割器",
        "滚柱直线导轨", "机械密封组件", "星形卸料阀", "皮带纠偏托辊", "离心泵叶轮", "工业脚踏开关",
        "膜片压力表", "旋风分离器模型", "双螺杆挤出机模型", "液压蓄能器模型",
    ],
    "wearable_accessories": [
        "机械怀表保护壳", "折叠眼镜鼻托盒", "皮革袖扣收纳卷", "黄铜领带夹展示座", "木质帽撑", "手摇鞋楦扩张器",
        "珐琅胸针盒", "金属手镯定型棒", "旅行皮带卷架", "折叠靴撑", "机械戒指尺寸规", "木质假领展示架",
        "发条式领带卷收器", "皮革眼镜链盒", "胸花保水夹", "帽檐弧度定型器",
    ],
    "architectural_models": [
        "悬索桥锚碇模型", "旋转灯塔剖面模型", "谷仓桁架模型", "水塔支架模型", "折叠舞台桁架模型", "拱坝泄洪口模型",
        "天文台圆顶模型", "风车磨坊剖面模型", "索道站台模型", "船闸闸门模型", "钟楼齿轮剖模", "温室穹顶骨架模型",
        "木构斗拱模型", "升降桥机械模型", "圆形竞技场剖模", "海上钻井平台模型",
    ],
}


COLORS = ["钴蓝色", "砖红色", "墨绿色", "暖灰色", "象牙白色", "琥珀色", "深紫色", "沙金色"]
MATERIALS = ["磨砂金属", "上釉陶瓷", "天然木质", "半透明玻璃", "细纹皮革", "阳极氧化铝", "编织纤维", "铸铁"]


def _all_subjects() -> list[tuple[str, str]]:
    rows = [(category, subject) for category, subjects in SUBJECT_CATALOG.items() for subject in subjects]
    if len(rows) != 304 or len({subject for _, subject in rows}) != 304:
        raise RuntimeError("subject catalog must contain exactly 304 unique subjects")
    return rows


def build_catalogs() -> tuple[list[dict], list[dict]]:
    development: list[dict] = []
    final: list[dict] = []
    for index, (category, subject) in enumerate(_all_subjects()):
        color = COLORS[index % len(COLORS)]
        material = MATERIALS[(index * 3) % len(MATERIALS)]
        prompt = (
            f"单个{color}{material}{subject}，保留主体身份、所有关键部件、连接关系与开口结构，"
            "物体完整居中且四周留有边距，三分之四产品视角，浅灰纯色背景，柔和均匀光照，无文字无人无其他物体"
        )
        if index < 240:
            split = "train" if index < 180 else "validation" if index < 210 else "dev_test"
            group_id = f"lora_{split}_{index:03d}"
            target = development
        else:
            split = "final_test"
            group_id = f"lora_final_{index - 240:03d}"
            target = final
        target.append(
            {
                "schema_version": PROMPT_SCHEMA_VERSION,
                "prompt_group_id": group_id,
                "prompt_id": group_id,
                "split": split,
                "category": category,
                "subject_key": subject,
                "subject_zh": f"{color}{material}{subject}",
                "prompt_zh": prompt,
                "prompt": prompt,
            }
        )
    validate_prompt_catalog(
        development,
        expected_split_counts={"train": 180, "validation": 30, "dev_test": 30},
    )
    validate_prompt_catalog(final, expected_split_counts={"final_test": 64})
    return development, final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "examples" / "sana_lora"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development, final = build_catalogs()
    write_jsonl_atomic(args.output_dir / "prompts240.jsonl", development)
    write_jsonl_atomic(args.output_dir / "final_test_prompts64.jsonl", final)
    print(
        json.dumps(
            {
                "development_groups": len(development),
                "final_test_groups": len(final),
                "semantic_subjects": len({row["subject_key"] for row in development + final}),
                "categories": len({row["category"] for row in development + final}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
