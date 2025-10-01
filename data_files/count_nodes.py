import networkx as nx
import os


def count_graphml_nodes(graphml_path):
    """
    计算GraphML文件中的节点数量
    :param graphml_path: GraphML文件的路径
    :return: 节点数量（整数）
    """
    # 检查文件是否存在
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML文件不存在：{graphml_path}")

    # 读取GraphML文件
    G = nx.read_graphml(graphml_path)

    # 获取节点数量（networkx图对象的内置方法）
    node_count = G.number_of_nodes()

    return node_count


# 示例：计算你生成的GraphML文件的节点数
if __name__ == "__main__":
    # 替换为你的GraphML文件路径（例如你之前生成的aes_secworks_orig.bench.graphml）
    graphml_file = "openabcd/graphml/aes_secworks_orig.graphml"

    # 计算并打印节点数
    try:
        count = count_graphml_nodes(graphml_file)
        print(f"GraphML文件中的节点数量：{count}")
    except Exception as e:
        print(f"错误：{e}")