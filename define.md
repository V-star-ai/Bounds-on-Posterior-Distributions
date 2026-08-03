## 数学定义

### 共同约定

本文中的所有分布都是子概率分布：总质量不要求等于1，但在数学语义上要求所有质量参数非负，并且总质量有限。实现中由于质量参数可能是求解器变量（Object），只在能静态判断时做非负性和收敛性检查；不能静态判断时保留表达式，不强行转为浮点数。

所有断点序列中的元素都是确定的`Fraction`。区间统一写作闭区间$[a,b]$；不同连续区间在端点处的重叠对Lebesgue连续部分无影响。若$a=b$，该区间表示一个狄拉克点质量所在的位置。一个高维区域中允许部分维度退化为狄拉克维度。

### MUD

MUD (Mixture Uniform Distribution)是一种有界的连续子概率分布，由多个均匀分布和狄拉克分布组成。

一个MUD记为$\text{MUD}(\mathcal{S}, P)$，其中$\mathcal{S} = \{S_1, S_2, \ldots, S_N\}$是一个分数序列集合，每个$S_i = (s_{i,1}, s_{i,2}, \ldots, s_{i,n_i + 1})$是一个不降的分数（有理数）序列（$s_{i,j} \leq s_{i,j+1}$），$P \in \R^{n_1 \times \cdots \times n_N}$是一个$N$维实数张量。允许某个$S_i$只有一个断点，此时$n_i=0$，表示该维没有任何区间；只要存在某个$n_i=0$，该MUD就是空测度。

对于每个维度$i$，$S_i$将该维度分割成了$n_i$段，对应与$P$中该维度的$n_i$个取值，$P$中的值代表该区域的概率质量。也就是说，对于$\vec{x}=\{x_1,\ldots,x_N\}$，对应了一个区域为$[s_{1,x_1}, s_{1, x_1 + 1}] \times \cdots \times [s_{N,x_N}, s_{N,x_N + 1}]$上质量为$P_\vec{x}$的一个均匀分布（因为$s_{i,j}$可能等于$s_{i,j+1}$，所以这个均匀分布可能在某些维度上退化为狄拉克分布，但仍是良定义的）。

更精确地，$P_\vec{x}$表示分配到该矩形块上的总质量，而不是密度值。若该块在某些维度长度为正，则在这些维度上按长度均匀分布；若某些维度长度为0，则在这些维度上为对应端点的狄拉克分布。MUD的总质量为$\sum_{\vec{x}} P_\vec{x}$。

MUD的单维截断$\text{restrict}(i, \text{op}, c)$表示乘上指示函数$\mathbf{1}[x_i\ \text{op}\ c]$，不做归一化。正长度区间按保留长度比例缩放质量；Dirac区间按端点是否满足条件整体保留或删除。若截断结果为空，则使用规范空表示：被截断维度的断点序列为$(c)$，其他维度只保留原左右端点$(s_{j,1}, s_{j,n_j+1})$，对应$P$张量在被截断维度上的长度为0。

代码实现中，MUD的网格结构已经抽象为`GridMUD + CellOps`，实现位于`distributions/mud.py`。当前数学意义上的MUD对应`MassMUD`，其cell payload是总质量，使用`MassCellOps`解释。为了兼容已有接口，代码中`MUD`仍作为`MassMUD`的别名。`align(target_S)`要求目标断点逐维包含源断点；若源中显式存在Dirac区间，目标中也必须显式包含对应的重复断点，否则不能精确保留点质量。

为支持后续“变量加均匀分布”的精确中间结果，代码中新增`AffineCell`、`AffineCellOps`和`AffineMUD`。`AffineCell(left,right,sloped=False)`表示某个cell在指定`affine_dim`上为一次函数密度，并用该cell左右端点处的函数值存储；`sloped=True`是几何斜坡标记，表示该cell在`affine_dim`上可能不是平台段。这个标记由卷积分段形状决定，不依赖`left == right`的静态判断，因此适用于求解器符号变量。非`affine_dim`维度仍按均匀质量比例解释。`AffineMUD.mass()`对`affine_dim`做梯形积分；`align`和`restrict`在`affine_dim`上做线性插值，在其他维度上按保留比例缩放端点值，并通过`AffineCellOps`传播`sloped`标记。`AffineMUD`不是替代`MassMUD`的新分布语义，而是卷积中的精确临时表示；调用`to_mass_mud_upper()`可用端点最大值把一次函数密度上近似回`MassMUD`，也可以传入约束变量工厂生成“新变量大于等于左右端点”的求解器约束。若传入`max_interval`，则在上近似前只对`sloped=True`且长度超过`max_interval`的区间做等长细分，从而得到更紧的上界；平台段不会被细分。

`MassMUD.convolve_uniform(dim, low, high)`表示对第`dim`维变量加上独立的`Uniform[low, high]`采样，其中要求`low < high`且均匀分布总质量为1。该操作不归一化，并精确返回`AffineMUD`。对源cell在卷积维的区间$[a,b]$和质量$m$：若$a<b$，输出密度为
$$
g(y)=\frac{m}{(b-a)(high-low)}\cdot \left|[a,b]\cap[y-high,y-low]\right|;
$$
若$a=b$，则Dirac质量卷积成区间$[a+low,a+high]$上的常数密度$m/(high-low)$。输出卷积维断点由所有$a+low,a+high,b+low,b+high$合并去重得到，因此不会产生新的Dirac区间。Dirac源区间产生的跳跃密度按目标cell的单侧语义存储，不把边界端点值泄漏到相邻cell。

`BGD.convolve_uniform(dim, low, high)`表示对BGD的第`dim`维加上独立`Uniform[low, high]`。新的中心块在该维的全局范围为$[A_i+low,B_i+high]$，左右扩展周期长度和衰减率保持不变。构造每个新块时，枚举所有卷积后与该新块有正长度交集的旧块：新中心块会接收旧中心块和左右尾块的贡献；新左块只接收旧左尾块贡献；新右块只接收旧右尾块贡献。由于该维卷积后没有Dirac质量，若旧块卷积结果与新块只在边界点相交，则视为没有贡献。对于`MassMUD` family，每个贡献先以`AffineMUD`精确累加，再通过`to_mass_mud_upper()`转成`MassMUD`上界；`max_interval`和`bound_factory`只属于这条上近似路径。对于`PolynomialMUD` family，每个贡献直接作为精确分段多项式累加并返回PolynomialBGD，不经过AffineMUD或MassMUD。

`add_constant(dim, c)`表示对第`dim`维做确定性平移$x_i:=x_i+c$。MUD层面直接把该维所有断点加上$c$。BGD层面由于只有中心块使用全局坐标，非中心块始终使用从0开始的局部坐标，因此只需要把中心块$E_{2,\ldots,2}$在第`dim`维的断点整体加上$c$；左右边缘块、周期长度、衰减率和质量张量均保持不变。对于一维MUD/BGD，`dist + c`和`c + dist`作为便捷写法等价于`add_constant(0,c)`或`shift(0,c)`。

例如：$\text{MUD}(\{(1,2,2), (0,1,3)\}, \begin{bmatrix}0.3 & 0.5 \\ 0.1 & 0.1\end{bmatrix})$，就由四部分组成：i. $[1,2] \times [0,1]$上的均匀分布，质量为0.3；ii. $[1,2] \times [1,3]$上的均匀分布，质量为0.5；iii. $[2,2] \times [0,1]$上的均匀分布，质量为0.1；iv. $[2,2] \times [1,3]$上的均匀分布，质量为0.1。iii和iv上在第一维上是狄拉克分布，在第二维上是均匀分布。

### BGD

我们定义BGD (Block Geometric Distribution) 是一种连续子概率分布。由一个MUD张量组成。

代码实现中，BGD主体位于`distributions/bgd.py`，并从`distributions/mud.py`导入MUD类型、区间工具和上近似所需的AffineMUD能力。为了兼容旧调用，`distributions.bgd`仍会重新导出`MUD`、`MassMUD`、`AffineMUD`等MUD层名字。

一个BGD记为$\text{BGD}(E, \vec{\alpha}, \vec{\beta})$，其中E是一个$\text{MUD}^{3^N}$的MUD张量，最中心的元素$E_{2,2,\ldots}$就是中心块。为了表述方便，记$C:=E_{2,2,\ldots}$。$\vec{\alpha}$和$\vec{\beta}$是一个$\R^N$的向量，分别表示左右衰减率。直观的讲，BGD的中心是一个MUD，其边缘是一圈MUD，再向外是以E中的MUD乘以离散的衰减率进行填充，整体是一个无限连续子概率分布。中心块$C$使用全局坐标，不要求从0开始；非中心边缘块使用局部坐标，并且每个维度的断点首项必须为0。对于每一个变量i，E中索引为1（负方向）的所有MUD的$s_{i,n_i+1}$相等，E中索引为3（正方向）的所有MUD的$s_{i,n_i+1}$相等。负方向和正方向的边缘长度不要求相等。

更精确地，设维度数为$N$。记中心块$C$在第$i$维的左右端点为$A_i=s^C_{i,1}$和$B_i=s^C_{i,n_i+1}$，中心长度为$L^C_i=B_i-A_i$。对每个非中心方向$\vec{d}\in\{-1,0,1\}^N\setminus\{\vec{0}\}$，用张量索引$\vec{d}+\vec{2}$表示对应的$E$元素，其中$-1,0,1$分别对应文中索引$1,2,3$。$E_{\vec{d}+\vec{2}}$是一个$N$维MUD。

对每个维度$i$，所有满足$d_i=-1$的边缘MUD在第$i$维必须具有相同的最右端点，记为负方向边缘块长度$L^-_i$；所有满足$d_i=1$的边缘MUD在第$i$维必须具有相同的最右端点，记为正方向边缘块长度$L^+_i$。$L^-_i$和$L^+_i$不要求相等。由于实现中会把这些边缘MUD作为从0开始的局部块使用，因此要求所有非中心$E$元素在每个维度上的局部断点首项为0。当$d_i=0$时，$E_{\vec{d}+\vec{2}}$在第$i$维使用中心块长度$L^C_i$；当$d_i=-1$时，使用负方向边缘块长度$L^-_i$；当$d_i=1$时，使用正方向边缘块长度$L^+_i$。

BGD可以用整数块坐标$\vec{k}\in\mathbb{Z}^N$描述。$\vec{k}=\vec{0}$对应中心块$C=E_{2,2,\ldots}$。当$\vec{k}\ne\vec{0}$时，定义方向
$$
d_i=\begin{cases}
-1,& k_i<0,\\
0,& k_i=0,\\
1,& k_i>0,
\end{cases}
$$
并使用边缘分布$E_{\vec{d}+\vec{2}}$。该块的衰减因子为
$$
g(\vec{k})=\prod_{i:k_i<0}\alpha_i^{-k_i-1}\prod_{i:k_i>0}\beta_i^{k_i-1}.
$$
因此与中心相邻的第一圈边缘块（$k_i\in\{-1,0,1\}$且$\vec{k}\ne\vec{0}$）衰减因子为1；继续向左/下等负方向移动时乘以对应$\alpha_i$，继续向右/上等正方向移动时乘以对应$\beta_i$。

块坐标到真实坐标的平移规则逐维定义。对第$i$维，令
$$
t_i(k_i)=\begin{cases}
A_i+k_i L^-_i,& k_i<0,\\
A_i,& k_i=0,\\
B_i+(k_i-1)L^+_i,& k_i>0.
\end{cases}
$$
中心块第$i$维覆盖$[A_i,B_i]$。非中心块$\vec{k}$在第$i$维覆盖$[t_i(k_i), t_i(k_i)+L^C_i]$（当$k_i=0$）、$[t_i(k_i), t_i(k_i)+L^-_i]$（当$k_i<0$）或$[t_i(k_i), t_i(k_i)+L^+_i]$（当$k_i>0$）。非中心$E$中的MUD始终按从0开始的局部坐标定义，放入全局BGD时再按$\vec{t}(\vec{k})$平移；中心块$C$已经使用全局坐标。

数学上要求$0\le \alpha_i<1$且$0\le \beta_i<1$，以保证无限几何填充的总质量有限。实现中若$\alpha_i,\beta_i$是求解器变量，则生成或保留相应约束，而不是试图数值判断。

BGD的局部块质量语义如下：中心块直接使用$E_{2,2,\ldots}$的质量；非中心块$\vec{k}$使用$g(\vec{k})$乘以$E_{\vec{d}+\vec{2}}$所表示的MUD质量。也就是说，$E$中的数值是第一圈对应方向块的基准质量，外圈质量由几何衰减得到。

#### BGD标准化

由于MUD允许退化区间，一个BGD在相邻块边界上可能用多种方式表示同一个单点质量。例如在一维中，若中心块右端点有一个Dirac质量，同时右侧扩展块的左端点也有Dirac质量，则这两个质量在全局坐标中落在同一点。右侧扩展块在继续向右几何平移时，如果左右端点都有Dirac质量，也会使同一族边界点质量被相邻块重复表示。BGD标准化的目标是在不改变分布测度的前提下，把这一族等价表示规约到同一个形式。

标准化只移动边界Dirac质量，不改变正长度区间上的均匀质量。对任意非中心方向$\vec{d}\in\{-1,0,1\}^N\setminus\{\vec{0}\}$和任意维度$i$：

- 若$d_i=-1$，检查$E_{\vec{d}+\vec{2}}$在第$i$维的最右侧是否存在Dirac区间，也就是其第$i$维断点序列最后两个断点相等，$s_{i,n_i}=s_{i,n_i+1}=L^-_i$。若存在，则把$P$张量中第$i$维下标为$n_i$的整片边界质量从$E_{\vec{d}+\vec{2}}$的右端点移除。它在靠近中心一侧的相邻块中应表示为第$i$维左端点质量：令$\vec{d}'$与$\vec{d}$只在第$i$维不同，且$d'_i=0$，把该边界切片加到$E_{\vec{d}'+\vec{2}}$的第$i$维左端点；若$\vec{d}'=\vec{0}$，则这正是加到中心块$E_{2,2,\ldots}$的第$i$维左端点。同时，为了保持继续向负方向的无限几何尾部不变，还需要把该边界切片乘以$\alpha_i$后加回$E_{\vec{d}+\vec{2}}$自身的第$i$维左端点。
- 若$d_i=1$，检查$E_{\vec{d}+\vec{2}}$在第$i$维的最左侧是否存在Dirac区间，也就是$s_{i,1}=s_{i,2}=0$。若存在，则把$P$张量中第$i$维下标为1的整片边界质量从$E_{\vec{d}+\vec{2}}$的左端点移除。它在靠近中心一侧的相邻块中应表示为第$i$维右端点质量：令$\vec{d}'$与$\vec{d}$只在第$i$维不同，且$d'_i=0$，把该边界切片加到$E_{\vec{d}'+\vec{2}}$的第$i$维右端点；若$\vec{d}'=\vec{0}$，则这正是加到中心块$E_{2,2,\ldots}$的第$i$维右端点。同时，为了保持继续向正方向的无限几何尾部不变，还需要把该边界切片乘以$\beta_i$后加回$E_{\vec{d}+\vec{2}}$自身的第$i$维右端点。
- 若$d_i=0$，该维不触发边缘向中心方向的标准化移动。

这里“加到左端点/右端点”表示：若目标MUD在对应边界已有Dirac区间，则把整片张量质量加到该区间；若没有，则在对应边界创建一个退化区间。若源和目标在其他维度的断点不同，需要先对齐到共同断点，再移动边界切片。

实现中`BGD.standardize(skip_static_zero=True)`默认会跳过整片质量都能静态判定为0的边界Dirac切片，以避免普通数值运算中引入无意义的零结构。若该BGD作为模板形状使用，应调用`standardize(skip_static_zero=False)`：此时即使边界切片当前质量为静态0，也会按上述规则做结构标准化，从而在后续创建求解器变量前保留必要的边界Dirac位置。

由于$E$中的每个元素都是MUD，边界切片的移除和加入都在MUD类内封闭，因此该标准化操作可以精确实现，不需要为保持独立性额外做上近似。

#### BGD截断

BGD的单维截断$\text{restrict}(i,\text{op},c)$表示乘上指示函数$\mathbf{1}[x_i\ \text{op}\ c]$，不做归一化。这里$c$是全局坐标。该操作通过重构中心块和边缘周期，可以在BGD表示内精确实现。

设第$i$维中心块左右端点为$A_i,B_i$。若截断点$c$落在中心块内，则直接截断所有$d_i=0$的MUD；对于$x_i>c$或$x_i\ge c$，负方向尾部置为空、正方向尾部保留；对于$x_i<c$或$x_i\le c$则反之。

若$c>B_i$且执行$x_i>c$或$x_i\ge c$，截断点落在正方向扩张区。设$c=B_i+qL^+_i+r$，其中$0\le r<L^+_i$。新的正方向周期由两部分拼接而成：当前周期的$[r,L^+_i]$部分乘以$\beta_i^q$，下一周期的$[0,r]$部分乘以$\beta_i^{q+1}$；新的衰减率仍为$\beta_i$。中心方向和负方向在该维置为空。

若$c>B_i$且执行$x_i<c$或$x_i\le c$，则正方向从中心到$c$之间的有限多个周期块被展开并并入$d_i=0$方向层，正方向无限尾部置为空，负方向尾部保留。

若$c<A_i$，负方向的处理与正方向对称：对于$x_i<c$或$x_i\le c$，重构负方向周期起点且衰减率仍为$\alpha_i$；对于$x_i>c$或$x_i\ge c$，从$c$到中心之间的有限负方向块被展开并并入$d_i=0$方向层，负方向无限尾部置为空。

高维情况下，只截断一个维度$i$。固定其他维度的方向后，沿第$i$维的$E_{\ldots,1,\ldots}$、$E_{\ldots,2,\ldots}$、$E_{\ldots,3,\ldots}$三块构成一条一维BGD链；截断逐条链独立完成。严格和非严格不等式的差异只影响Dirac质量，由MUD的截断规则处理。构造完成后进行标准化，消除边界Dirac质量的重复表示。

#### BGD框架对齐

BGD框架对齐用于在不改变原分布测度的前提下，把一个BGD重表达为另一个中心域或周期结构。它是实现BGD加法前的基础操作。

`align_center_domain(lefts, rights)`把BGD精确重表达到新的中心矩形$\prod_i[A'_i,B'_i]$上。要求新中心包含旧中心，即$A'_i\le A_i$且$B'_i\ge B_i$。如果某一维向左扩展中心，则旧左侧尾部中落入$[A'_i,A_i]$的有限多个周期块被展开并并入$d_i=0$方向层，剩余左侧无限尾部以$A'_i$为新的周期边界重构相位；向右扩展时对称处理。该操作保持$\vec\alpha,\vec\beta,L^-_i,L^+_i$不变，并保持总测度不变。

`align_edge_periods(left_lengths, right_lengths)`把左右扩展周期精确放大到目标长度。要求每个目标长度都是当前对应周期长度的正整数倍。若$L_i^{-,*}=m_i^-L_i^-$，则新的左侧第一周期块由$m_i^-$个旧左侧周期块拼接而成，从远离中心到靠近中心的质量分别乘以$\alpha_i^{m_i^- -1},\ldots,\alpha_i,1$，新的左侧衰减率为$(\alpha_i)^{m_i^-}$。若$L_i^{+,*}=m_i^+L_i^+$，则新的右侧第一周期块由$m_i^+$个旧右侧周期块拼接而成，从靠近中心到远离中心的质量分别乘以$1,\beta_i,\ldots,\beta_i^{m_i^+-1}$，新的右侧衰减率为$(\beta_i)^{m_i^+}$。高维方向块按各扩展维度的有限笛卡尔积展开，衰减因子相乘。该操作保持总测度不变。

#### BGD加法

`BGD + BGD`表示两个子概率测度的直接加法，不是卷积，也不做归一化。实现中先把两个BGD对齐到共同框架，再逐方向相加。

共同中心域取两个中心矩形的最小包围矩形：
$$
A_i^*=\min(A_i^{(1)},A_i^{(2)}),\quad B_i^*=\max(B_i^{(1)},B_i^{(2)}).
$$

共同左右扩展周期取有理数意义下的最小公倍长度。对正有理数$\ell_1,\ell_2$，$\operatorname{lcm}_{\mathbb Q}(\ell_1,\ell_2)$定义为最小正有理数$\ell^*$，使得$\ell^*/\ell_1$和$\ell^*/\ell_2$都是正整数。若$\ell_j=a_j/b_j$为既约分数，则多项有理lcm可由“分子整数lcm / 分母整数gcd”得到。

两个输入分别执行`align_center_domain`和`align_edge_periods`后，中心域和周期长度完全一致。此时若对应衰减率仍不相同，结果衰减率逐维取可比较数值的最大值：
$$
\alpha_i^*=\max(\alpha_i^{(1)},\alpha_i^{(2)}),\quad
\beta_i^*=\max(\beta_i^{(1)},\beta_i^{(2)}).
$$
把衰减率较小的一方通过`relax_decay`提升到共同衰减率，得到对原BGD的上界；若两个衰减率相同，则该步骤是精确的。最后对每个方向块执行MUD加法：
$$
E^*_{\vec d}=E^{(1)}_{\vec d}+E^{(2)}_{\vec d}.
$$
因此当对齐后的衰减率相同，`BGD + BGD`是精确测度加法；当衰减率不同，结果是逐点不小于真实和的BGD上界。当前实现中默认自动取`max`只支持可静态比较的数值衰减率；若衰减率是求解器符号，可以通过`add(other, max_fn=...)`传入外部max函数生成共同衰减率表达式。此时实现信任外部max函数满足上界约束，即生成的结果应满足$\alpha_i^*\ge\alpha_i^{(1)},\alpha_i^{(2)}$和$\beta_i^*\ge\beta_i^{(1)},\beta_i^{(2)}$。

#### 独立联合分布

`independent_product`用于把两个相互独立的分布组合为联合分布。若$G_1$是$N_1$维BGD，$G_2$是$N_2$维BGD，则
$$
G=G_1\otimes G_2
$$
是一个$N_1+N_2$维BGD，并满足
$$
\mu_G(A\times B)=\mu_{G_1}(A)\mu_{G_2}(B),\quad
\text{mass}(G)=\text{mass}(G_1)\text{mass}(G_2).
$$

维度顺序固定为先保留$G_1$的所有维度，再追加$G_2$的所有维度。MUD层面的独立乘积为断点拼接和质量张量外积：
$$
\text{MUD}(\mathcal S^{(1)},P^{(1)})\otimes
\text{MUD}(\mathcal S^{(2)},P^{(2)})
=
\text{MUD}(\mathcal S^{(1)}\Vert\mathcal S^{(2)},P),
$$
其中
$$
P_{\vec i,\vec j}=P^{(1)}_{\vec i}P^{(2)}_{\vec j}.
$$

BGD层面的独立乘积逐方向拼接：
$$
E_{\vec d^{(1)}\Vert\vec d^{(2)}}=
E^{(1)}_{\vec d^{(1)}}\otimes E^{(2)}_{\vec d^{(2)}},
$$
并令
$$
\vec\alpha=\vec\alpha^{(1)}\Vert\vec\alpha^{(2)},\quad
\vec\beta=\vec\beta^{(1)}\Vert\vec\beta^{(2)}.
$$
只有联合中心块$E_{\vec 0\Vert\vec 0}$使用两个输入中心块的全局坐标；如果某个输入的中心块出现在联合的非中心方向块中，则该中心块会转换为从0开始的局部坐标，以满足非中心边缘块的坐标约定。该操作在BGD表示内精确封闭，不需要上近似。

#### 维度边缘化与重新赋值

`marginalize(dim)`表示对第`dim`维做边缘化。MUD层面直接删除该维断点，并对质量张量沿该轴求和。BGD层面需要把被删除维度的三个方向合并：中心方向只出现一次，负方向和正方向分别对应无限几何尾部，因此对剩余方向$\vec d_{\neg i}$有
$$
E'_{\vec d_{\neg i}}=
\operatorname{marg}_i(E_{\ldots,-1,\ldots})\frac{1}{1-\alpha_i}
+\operatorname{marg}_i(E_{\ldots,0,\ldots})
+\operatorname{marg}_i(E_{\ldots,1,\ldots})\frac{1}{1-\beta_i}.
$$
若结果块是中心块，而某个贡献来自原来的非中心块，则剩余的中心方向坐标需要从局部坐标平移回全局中心坐标。该操作保持总质量不变。若一维BGD边缘化掉唯一维度，结果退化为标量总质量。

`permute_dims(order)`表示维度重排。MUD层面重排断点序列并转置质量张量；BGD层面同时重排$E$张量轴、每个MUD内部维度、$\vec\alpha$和$\vec\beta$。

`replace_dim(dim, new_bgd)`表示把原BGD的第`dim`维替换为一个新的一维BGD，并令新维度与其他维度独立。其精确定义为
$$
\operatorname{replace\_dim}(G,i,H)
=
\operatorname{permute}\left(\operatorname{marginalize}_i(G)\otimes H\right),
$$
其中$H$必须是一维BGD，维度顺序保持为原位置不变。替换后总质量为
$$
\text{mass}(\operatorname{replace\_dim}(G,i,H))=\text{mass}(G)\text{mass}(H).
$$
因此如果希望替换不改变总质量，需要额外保证$\text{mass}(H)=1$；实现本身不做归一化。该操作在BGD表示内精确封闭，不需要上近似。

#### BGD小于等于约束

`le_constraints(other)`用于生成足以保证两个BGD满足逐点测度上界关系`self <= other`的约束列表。实现先把两个BGD精确对齐到共同框架：中心域取最小包围矩形，左右扩展周期取有理lcm。随后生成两类约束：

- 对每个维度$i$，生成$\alpha_i^{self}\le \alpha_i^{other}$和$\beta_i^{self}\le \beta_i^{other}$。在非负质量和非负衰减率语义下，这保证尾部几何因子逐层不超过右侧BGD。
- 对每个方向块$E_{\vec d}$，先把左右两个MUD对齐到共同断点，再对每个对应cell生成质量约束$P^{self}_{\vec x}\le P^{other}_{\vec x}$。对齐后两个cell表示同一个矩形或Dirac区域，因此该约束保证局部块测度上界。

默认返回三元组`(left, "<=", right)`；若传入`constraint_factory(left, right, name)`，则返回该工厂生成的求解器约束对象。该接口只生成约束，不做可满足性检查，也不对求解器符号做数值判断。

例如一个二维的例子：
$$
\begin{matrix}
\alpha_1^2 \beta_2^2 \times E_{1,3} & \alpha_1 \beta_2^2 \times E_{1,3} & \beta_2^2 \times E_{1,3} & \beta_2^2 \times E_{2,3} & \beta_2^2 \times E_{3,3} & \beta_1 \beta_2^2 \times E_{3,3} & \beta_1^2 \beta_2^2 \times E_{3,3} \\
\alpha_1^2 \beta_2 \times E_{1,3} & \alpha_1 \beta_2 \times E_{1,3} & \beta_2 \times E_{1,3} & \beta_2 \times E_{2,3} & \beta_2 \times E_{3,3} & \beta_1 \beta_2 \times E_{3,3} & \beta_1^2 \beta_2 \times E_{3,3} \\
\alpha_1^2 \times E_{1,3} & \alpha_1 \times E_{1,3} & E_{1,3} & E_{2,3} & E_{3,3} & \beta_1 \times E_{3,3} & \beta_1^2 \times E_{3,3} \\
\alpha_1^2 \times E_{1,2} & \alpha_1 \times E_{1,2} & E_{1,2} & C & E_{3,2} & \beta_1 \times E_{3,2} & \beta_1^2 \times E_{3,2} \\
\alpha_1^2 \times E_{1,1} & \alpha_1 \times E_{1,1} & E_{1,1} & E_{2,1} & E_{3,1} & \beta_1 \times E_{3,1} & \beta_1^2 \times E_{3,1} \\
\alpha_1^2 \alpha_2 \times E_{1,1} & \alpha_1 \alpha_2 \times E_{1,1} & \alpha_2 \times E_{1,1} & \alpha_2 \times E_{2,1} & \alpha_2 \times E_{3,1} & \beta_1 \alpha_2 \times E_{3,1} & \beta_1^2 \alpha_2 \times E_{3,1} \\
\alpha_1^2 \alpha_2^2 \times E_{1,1} & \alpha_1 \alpha_2^2 \times E_{1,1} & \alpha_2^2 \times E_{1,1} & \alpha_2^2 \times E_{2,1} & \alpha_2^2 \times E_{3,1} & \beta_1 \alpha_2^2 \times E_{3,1} & \beta_1^2 \alpha_2^2 \times E_{3,1} \\

\end{matrix}
$$

其中第一维度和第二维度的正方向分别是右和上（经典直角坐标系）。



## 代码实现

使用python实现。对于上述的所有分布，断点$\mathcal{S}$中的元素都是确定的Fraction。而$P,\vec{\alpha}, \vec{\beta}$ 等实数参数内可能是实数或求解器变量（Object，支持基本运算），实现时需要注意兼容Object。



BGD支持如下运算：

- [x] standardize()标准化边界Dirac质量表示
- [x] BGD + BGD（函数上的直接加法，而不是卷积）
- [x] independent_product：组合两个独立BGD为一个联合分布BGD
- [x] restrict($x_i > c / x_i \ge c / x_i < c / x_i \le c$)等截断函数
- [x] 对某个维度重新赋值，赋值为一个新的一维BGD分布。该操作表示替换该维度的边缘分布；替换后的变量与其他维度独立。
- [x] 将某一个维度加上一个均匀分布（变量上的加法，简单的卷积）
- [x] 将某一个维度加上固定常数（确定性平移）
- [x] 建立两个BGD的关于某个约束函数的约束列表

这些运算不都在BGD类内封闭。BGD在本项目中的用途是表示上界；当精确结果不是BGD时，实现应构造一个BGD形式的上近似，并保持“结果分布逐点不小于真实结果”的上界语义。

### 逐步实现计划

1. 基础数据结构 [x]
   - 实现`GridMUD`、`MassMUD`、`MUD`兼容别名和`BGD`核心类。
   - 抽象`CellOps`接口；当前质量版本使用`MassCellOps`，cell payload解释为总质量。
   - 所有断点使用`Fraction`保存。
   - 所有质量、衰减率和中间表达式使用普通Python对象保存，避免强制转换为`float`。
   - 构造时检查维度、张量形状、断点不降、非中心边缘块断点从0开始、BGD边缘块长度规则等结构性不变量。

2. 公共区间与张量工具 [x]
   - 实现断点规范化、区间长度、区间交集、区间包含、Dirac区间判断等基础函数。
   - 实现断点合并工具，默认保留输入中显式出现的Dirac区间；必要时可以选择去重合并。
   - 实现按多维索引枚举张量区域的工具。
   - 实现对象安全的加法、乘法、求和工具，避免依赖数值类型专有接口。

3. MUD的核心能力 [x]
   - 实现`mass()`计算总质量。
   - 实现`align(target_S)`，把分布重分割到给定断点上。
   - 对普通正长度区间按长度比例分配质量；对Dirac区间，目标断点中必须显式包含对应的退化区间，否则不能在精确align中保留该点质量。
   - `align(target_S)`要求目标网格逐维包含源网格断点，避免目标区间跨过源区间边界导致cell payload语义不清。
   - 将`mass`、`scale`、`add`、`align`、`restrict`、`independent_product`、`marginalize`和`permute_dims`改为通过`CellOps`解释cell payload；现有`MassMUD`行为已通过一致性测试。

4. BGD的核心能力 [x]
   - 实现块方向、块衰减因子、块平移、块长度和`block_at(k)`。
   - `block_at(k)`返回该块的局部分布、全局平移和衰减因子。
   - 实现有限块区域枚举接口，用于测试和需要有限展开的操作。
   - 实现符号化总质量表达式：中心元素$E_{2,2,\ldots}$质量加上所有边缘方向的几何级数质量。

5. BGD标准化 [x]
   - 实现`standardize()`，把相邻块边界上的Dirac质量移动到规范位置。
   - 对源MUD和目标MUD先进行必要的断点对齐，再精确移动边界张量切片。
   - 标准化后每个边缘块仍是MUD，因此该操作在BGD表示内封闭。

6. restrict截断 [x]
   - 先实现MUD在单维约束$x_i > c$、$x_i \ge c$、$x_i < c$、$x_i \le c$下的精确截断。
   - MUD截空时使用规范空表示：截断维度为$(c)$，其他维度保留左右端点。
   - 再实现BGD截断：中心内截断直接处理；扩张区截断通过重构周期起点或展开有限前缀精确处理。

7. BGD + BGD [x]
   - [x] 实现`align_center_domain(lefts, rights)`：把中心域扩展到目标矩形，并精确重构周期相位。
   - [x] 实现`align_edge_periods(left_lengths, right_lengths)`：把扩展周期放大到整数倍长度，并把衰减率提升为对应倍数次方。
   - [x] 实现有理周期长度的lcm工具，用于为两个BGD选择共同左右扩展周期。
   - [x] 实现`relax_decay(alpha, beta)`：把衰减率放大到目标值，得到上界BGD。
   - [x] 实现完整`BGD + BGD`：中心域取最小包围矩形，扩展周期取lcm，对齐后衰减率取max，再逐方向MUD相加。
   - [x] 支持`add(other, max_fn=...)`传入外部max函数，以适配求解器符号衰减率。
   - [x] 后续约束列表生成时，应为外部max函数产生的共同衰减率补充上界约束。

8. 独立联合分布 [x]
   - [x] 实现`MUD.independent_product(other)`：断点拼接，质量张量做外积。
   - [x] 实现`BGD.independent_product(other)`：逐方向块做MUD独立乘积，衰减率向量拼接。
   - [x] 正确处理中心块坐标：只有联合中心使用全局坐标，非中心块中的输入中心块转换为局部坐标。
   - [x] 覆盖Dirac、Object质量、一维到二维、以及一维到二维再拼接为三维的测试。

9. 维度重新赋值 [x]
   - [x] 实现`MUD.marginalize(dim)`和`MUD.permute_dims(order)`。
   - [x] 实现`BGD.scale(factor)`、`BGD.marginalize(dim)`和`BGD.permute_dims(order)`。
   - [x] 实现`BGD.replace_dim(dim, new_bgd)`：删除原分布中该维的依赖，将该维替换为给定的一维BGD。
   - [x] 替换后该维与其他维度独立，且总质量为原BGD质量乘以新一维BGD质量。
   - [x] 覆盖Dirac、维度重排、一维替换特例、Object质量和边缘化中心坐标平移测试。

10. 单维加均匀分布 [ ]

       - [x] 抽象`AffineCell`、`AffineCellOps`和`AffineMUD`，用于保存卷积后每个cell内的一次函数密度。
       - [x] 实现`AffineMUD`的`mass`、`align`、`restrict`、非affine维`marginalize`和`to_mass_mud_upper()`，并通过测试确认与`MassMUD`兼容。
       - [x] 先实现一维MUD与均匀分布的卷积：`MassMUD.convolve_uniform(dim, low, high)`精确返回`AffineMUD`。

       - [x] 实现`MassMUD.convolve_uniform_upper(...)`，将精确`AffineMUD`转换为`MassMUD`上界。

       - [x] 再提升到MUD/BGD的指定维度卷积：`BGD.convolve_uniform(dim, low, high, max_fn=None, bound_factory=None)`返回上界BGD，并支持约束变量路径。

       - [x] 实现`AffineCell.sloped`几何斜坡标记和`max_interval`细分上近似：只对斜坡段按最大长度细分，平台段保持不变。

11. 约束列表生成 [x]

       - [x] 定义约束函数接口。

       - [x] 为BGD生成求解器可消费的`self <= other`约束表达式列表：衰减率逐维约束，MUD块对齐后逐cell质量约束。

       - [x] 对求解器变量只组合表达式，不做数值判断。

12. 测试 [x, ongoing]
    - 从一维MUD、二维MUD、一维BGD、二维BGD逐步测试。
    - 覆盖普通区间、Dirac区间、Fraction断点、Object参数、align、restrict边界、几何衰减和上近似语义。


## 分段多项式BGD

### 目标与范围

分段多项式BGD将当前MUD中“每个cell内密度为常数”的表示推广为“每个cell内密度为多元多项式”。第一阶段只构造精确的符号语义和约束，不负责求解约束。语义层不得通过采样、端点比较、Bernstein系数比较或次数截断把区域多项式约束提前近似为有限标量约束；后续求解后端可以独立选择SOS/SDP、Bernstein证书或其他方法处理这些约束。

每个非退化cell
$$
C=[a_1,b_1]\times\cdots\times[a_N,b_N]
$$
使用局部归一化状态坐标
$$
u_i=\frac{x_i-a_i}{b_i-a_i}\in[0,1].
$$
cell payload是定义在局部坐标上的稀疏幂基多项式
$$
p_C(\vec u)=\sum_{\vec k}c_{\vec k}\vec u^{\vec k}.
$$
它表示相对于该cell的Lebesgue/Dirac混合基准测度的密度，而不是cell总质量。若第$i$维退化为$[a_i,a_i]$，该维使用$\delta_{a_i}$作为基准测度，并要求$p_C$不依赖$u_i$；质量计算时该维贡献因子为1。

### 参数多项式与状态多项式

需要严格区分两类符号：

- 参数变量$\theta_j$表示循环模板cell系数、BGD衰减率以及语义变换引入的辅助未知量。
- 状态变量$u_i$表示cell内被全称量化的局部程序变量。

参数多项式使用精确有理系数：
$$
c(\vec\theta)=\sum_{\vec m}q_{\vec m}\vec\theta^{\vec m},
\qquad q_{\vec m}\in\mathbb Q.
$$
状态多项式以参数多项式作为系数：
$$
p(\vec u;\vec\theta)
=
\sum_{\vec k}c_{\vec k}(\vec\theta)\vec u^{\vec k}.
$$
这个分层结构使积分、状态变量代换和维度操作只作用于$\vec u$，同时完整保留系数之间关于$\vec\theta$的多项式关系。实现中不允许除以含参数的表达式；若精确语义需要$q=p/d$且已知$d>0$，则引入新参数$q$并添加多项式等式$dq-p=0$。

### 多项式约束IR

语义构造阶段保留以下与求解器无关的约束：

- `ParameterConstraint`：只含参数变量的标量多项式等式或不等式。
- `DomainPolynomialConstraint`：状态多项式在指定单位盒上恒满足等式或不等式。
- `PolynomialIdentity`：两个多项式恒等，等价于对应系数多项式分别相等。

例如，对齐后的两个cell满足上界关系时，生成
$$
\forall\vec u\in[0,1]^D,\qquad
p_{\mathrm{right}}(\vec u;\vec\theta)
-p_{\mathrm{left}}(\vec u;\vec\theta)\ge0,
$$
其中$D$只包含非退化维度。该约束在语义层保持完整，不转换为有限采样点约束。

### PolynomialCell运算语义

`PolynomialCellOps`需要精确支持以下运算：

- `add`和`scale`：对稀疏多项式做精确代数运算。
- `restrict`和`align`：对子cell执行局部坐标仿射代换$u_{\mathrm{old}}=r+s u_{\mathrm{new}}$。
- `mass`：在所有非退化局部维度的$[0,1]$上积分，再乘对应物理区间长度。
- `independent_product`：拼接状态变量并相乘。
- `marginalize_dim`：对被删除的非退化维度积分；Dirac维直接删除。
- `permute_dims`：同时重排网格维度和多项式状态变量。
- `nonnegative_constraint`：生成cell密度在其单位盒上恒非负的区域约束。
- `le_constraint`：生成两个cell多项式之差在单位盒上恒非负的区域约束。

所有操作都使用`Fraction`保存确定系数，不执行浮点化、次数截断或未经证明的降阶。

### PolynomialBGD语义

BGD主体应重构为对`GridMUD`的cell family泛型，而不是复制一套PolynomialBGD算法。空块、边界Dirac切片、平移、标准化和框架对齐都必须通过原分布的`_new()`及`CellOps`构造，从而保留具体payload类型。

PolynomialBGD继续使用当前中心块、方向块和几何衰减语义。非中心块中的多项式使用块内局部归一化坐标；块在全局空间平移时多项式系数不变。只要局部多项式非负，增大$\alpha_i,\beta_i$仍然产生合法上界。

两个PolynomialBGD的上界约束先精确对齐中心域、周期和cell网格，再生成：

- 衰减率约束$\alpha_i^{L}\le\alpha_i^{R}$和$\beta_i^{L}\le\beta_i^{R}$。
- 每个对应cell上的区域多项式约束$p_R-p_L\ge0$。

### 均匀卷积

对第$i$维cell多项式密度$p(x_i)$与`Uniform[low,high]`卷积：
$$
g(y)=\frac{1}{high-low}
\int_{\max(a,y-high)}^{\min(b,y-low)}p(x)\,dx.
$$
用$a+low,a+high,b+low,b+high$划分目标区间后，每段积分上下限都是常数或$y$的仿射函数，因此结果仍为精确多项式，且该维次数至多增加1。实现应直接返回PolynomialMUD，不经过AffineMUD或MassMUD上近似。Dirac源cell卷积后成为连续维上的零次多项式密度。

### 循环约束

第一阶段循环只构造Park前不动点约束。对入口分布$\mu_0$、循环守卫$g$、循环体抽象变换$T$和符号PolynomialBGD模板$I$，保留
$$
\mu_0+T(\mathbf1_g I)\le I.
$$
最终符号结果为$\mathbf1_{\neg g}I$。循环构造返回符号PolynomialBGD、参数声明和完整多项式约束集合，不调用求解器。

循环约束不通过符号`max`先构造BGD和。实现应提供`leq_sum([lower_1,\ldots,lower_n], upper)`形式的约束生成：各lower衰减率分别不超过upper衰减率，对齐后的局部多项式贡献之和不超过upper局部多项式。这样所有保留的关系仍是多项式约束。

### 实现计划

1. 精确多项式与约束IR [x]
   - 实现参数变量、参数多项式和状态多项式。
   - 实现精确加减乘、非负整数次幂、有理常数除法、状态变量仿射代换、积分、维度拼接和维度重排。
   - 实现参数约束、单位盒区域多项式约束、恒等约束及约束构造上下文。
   - 为符号除法提供“新参数 + 多项式等式”的精确提升接口。

2. PolynomialCell与PolynomialMUD [x]
   - 实现`PolynomialCellOps`和`PolynomialMUD`。
   - 定义并验证连续维和Dirac维的混合密度语义。
   - 实现MassMUD到零次PolynomialMUD的精确嵌入。
   - 覆盖mass、align、restrict、independent_product、marginalize和permute_dims测试。

3. BGD泛型化 [x]
   - 移除BGD及其辅助函数中对`MassMUD/MUD`的硬编码。
   - 让标准化、边界Dirac移动、截断、框架对齐、加法和维度操作保留cell family。
   - 用现有MassMUD测试确认泛型化不改变当前数学语义。

4. PolynomialBGD区域约束 [x]
   - 让BGD非负约束和小于等于约束委托给`CellOps`。
   - 实现`leq_sum`，避免在循环约束中引入符号max。
   - 实现指定次数的符号PolynomialBGD模板构造。

5. 精确均匀卷积 [x]
   - 实现PolynomialMUD指定维度与Uniform分布的精确卷积。
   - 提升到PolynomialBGD并正确累加有限个相交源块。
   - 验证质量保持、次数增长、Dirac卷积和多维单轴卷积。

6. 无循环程序语义构造 [x]
   - 将无循环PolynomialBGD语义构造从Adapter求解中分离。
   - 接入精确多项式先验、赋值、条件、概率选择和observe。
   - 返回未归一化PolynomialBGD与`ConstraintProblem`，不调用Adapter。

7. 循环约束与求解器接入 [进行中]
   - [x] 实现实验性SCIP后端，直接编译有界参数多项式，并用有限次分解式SOS证书编译单位盒区域约束。
   - [x] 验证衰减率参数与cell多项式系数可以在同一个非凸模型中联合优化。
   - [x] 将SCIP注册为统一`Adapter`，接通旧MassBGD Park循环路径，并保留新多项式IR专用入口。
   - [x] 实现并测试PolynomialBGD的Park循环模板与不动点约束。
   - [x] 让循环构造返回符号PolynomialBGD、完整多项式约束问题和精确提升后的质量目标。
   - [x] 按有限probe逐变量推理模板次数，并提供统一升次实验开关。
   - [ ] 支持嵌套循环和循环体内的独立分布替换。

8. 约束检查与序列化 [ ]
   - 实现多项式和约束的稳定打印、结构统计及机器可读序列化。
   - 增加基于给定参数和状态点的约束求值功能，仅用于测试和调试，不作为区域约束证明方法。

### 当前多项式语义实现

第一阶段实现位于`semantics/polynomial.py`和`semantics/constraints.py`。

`ParameterVariable`按名字标识参数未知量。`ParameterPolynomial`使用规范化稀疏单项式表保存关于参数变量的精确多项式，自动合并同类项并删除零项；只允许除以非零有理常数。`StatePolynomial`使用固定长度的状态指数元组作为单项式索引，并以`ParameterPolynomial`作为系数，因此参数乘积和状态变量乘积都不会退化为通用Object表达式。

`StatePolynomial.affine_substitute(dim, offset, scale)`精确实现
$$
u_{dim}\mapsto offset+scale\cdot u_{dim},
$$
用于后续cell截断和对齐。`integrate_unit`、`antiderivative`、`independent_product`和`permute_dims`分别提供单位区间积分、单维原函数、独立维度拼接和状态维度重排。

约束实现包含`ParameterConstraint`、`DomainPolynomialConstraint`、`PolynomialIdentity`和`UnitBoxDomain`。`DomainPolynomialConstraint`会检查多项式不得依赖被标记为非活动的Dirac维度。`ConstraintContext`统一声明参数、生成无冲突辅助参数并收集约束；构建不可变`ConstraintProblem`时会拒绝任何未声明参数。`exact_positive_quotient`通过新参数$q$、正分母约束$d>0$和恒等式$dq-p=0$精确表示$q=p/d$，不在多项式IR中引入符号除法。

当前测试位于`tests/test_polynomial_semantics.py`，覆盖规范化、精确有理运算、参数与状态分层、仿射代换、积分、原函数、独立积、维度重排、单位盒约束、Dirac非活动维检查、多项式恒等和正分母辅助商。

第二阶段实现位于`distributions/polynomial_mud.py`。`PolynomialCell`封装一个`StatePolynomial`密度，`PolynomialCellOps`根据cell物理区间解释该密度。对连续维$[a_i,b_i]$，质量计算包含
$$
(b_i-a_i)\int_0^1du_i;
$$
对Dirac维$[a_i,a_i]$，质量因子为1，并在构造时禁止多项式依赖对应局部状态变量。

`PolynomialCellOps.restrict`和继承的`GridMUD.align`通过
$$
u_{\mathrm{source}}
=
\frac{a_{\mathrm{target}}-a_{\mathrm{source}}}
{b_{\mathrm{source}}-a_{\mathrm{source}}}
+
\frac{b_{\mathrm{target}}-a_{\mathrm{target}}}
{b_{\mathrm{source}}-a_{\mathrm{source}}}
u_{\mathrm{target}}
$$
精确重参数化密度，不做归一化。连续源cell不会向退化目标cell产生Dirac质量；Dirac源cell只会对显式相同的Dirac目标cell产生贡献。

`PolynomialMUD.marginalize`对删除维度的每个cell密度做单位区间积分并乘物理长度，然后在剩余网格cell上求和；删除Dirac维时不乘零长度。`independent_product`拼接两个多项式的局部状态变量，`permute_dims`同时重排网格张量和多项式指数轴。

`PolynomialMUD.from_mass_mud`把MassMUD的cell总质量$m_C$精确转换为零次密度
$$
p_C=\frac{m_C}{\prod_{i:b_i>a_i}(b_i-a_i)}.
$$
Dirac维不进入分母，因此该嵌入同时适用于连续、纯Dirac和混合cell，并保持每个cell及整个MUD的总质量。

`PolynomialCellOps.nonnegative_constraint`和`le_constraint`分别生成完整的单位盒区域非负约束；Dirac维在`UnitBoxDomain`中标记为非活动维。用于调试的`evaluate`先把物理坐标精确映射到cell局部坐标，再返回仍保留参数变量的`ParameterPolynomial`。

第二阶段测试位于`tests/test_polynomial_mud.py`，覆盖连续和混合Dirac质量、align和restrict仿射重参数化、空分布cell family保持、多项式加法和参数缩放、独立积、连续/Dirac边缘化、维度重排、MassMUD精确嵌入及区域约束构造。

第三阶段把`BGD`定义为对`GridMUD` cell family参数化的统一外层结构，而不是再增加一个与原类平行的`PolynomialBGD`类。构造时要求$3^N$个block都是`GridMUD`、维度均为$N$，并且属于同一个具体cell family；因此一个BGD不能隐式混合`MassMUD`与`PolynomialMUD`。不同family之间的显式嵌入仍由`PolynomialMUD.from_mass_mud`负责。

BGD辅助函数现在遵守如下类型保持不变量：从一个block切出或删除边界Dirac、把Dirac移动到新点、沿某维平移、构造零网格以及为截断构造空分布时，结果都通过源block的`_new()`和实例`ops.zero()`创建。`GridMUD.is_static_zero`把零值判定委托给`CellOps.is_static_zero`，因此标准化可以识别零多项式，同时不会假设cell payload是标量。

在此不变量下，`scale`、`standardize`、`restrict`、`align_center_domain`、`align_edge_periods`、`align_frame`、`add`、`independent_product`、`marginalize`、`permute_dims`和`replace_dim`都会保持`PolynomialMUD` family。边界Dirac标准化仍使用原BGD语义：靠近中心的Dirac质量移动到相邻块，并以对应$\alpha_i$或$\beta_i$缩放后回灌到下一周期；当衰减率是`ParameterPolynomial`时，该乘积仍精确保留为cell多项式的参数系数。

第三阶段没有改变多项式约束与卷积的数学接口；PolynomialBGD区域约束和精确均匀卷积现已分别在第四、第五阶段完成。

此外，有限次衰减缩放$\alpha_i^k,\beta_i^k$已经能作为参数多项式参与block操作；但无限尾质量中的$1/(1-\alpha_i)$和$1/(1-\beta_i)$是有理函数，不属于`ParameterPolynomial`。因此当前`mass`和删除尾方向的`marginalize`在衰减率为有理常数时是精确的；符号衰减率的尾和需要在后续语义构造阶段接入`ConstraintContext.exact_positive_quotient`，以辅助参数和多项式恒等式表示，不能直接在多项式IR中做除法。

第三阶段测试位于`tests/test_polynomial_bgd.py`，覆盖family构造不变量、跨family拒绝、精确参数缩放、截断重参数化、中心和周期框架对齐、Dirac边界标准化、符号衰减系数、加法、独立积、边缘化、维度重排以及尚未泛化接口的明确类型边界。

第四阶段为`CellOps`增加统一约束构造接口。标量质量cell仍可生成原有三元组或通过`constraint_factory`转换成旧求解器约束；`PolynomialCellOps`则直接生成语义IR，不接受求解器工厂。`BGD.nonnegative_constraints()`对每个维度生成
$$
0\le\alpha_i<1,\qquad 0\le\beta_i<1,
$$
并对每个非空cell生成其局部单位盒上的区域多项式非负约束。对于Dirac维，`UnitBoxDomain.active_dims`不包含该维，因此约束不会把不存在的连续自由度引入证明域。

`BGD.le_constraints(right)`现在等价于`leq_sum([left], right)`。`leq_sum(lowers, upper)`不先调用BGD加法，也不构造任何符号`max`。它先取所有BGD中心域的并和各方向周期长度的最小公倍数，把每个输入精确对齐并标准化，再合并对应block的cell网格。若共同周期是某输入原周期的$k$倍，则先由框架对齐把该输入衰减率变为$\alpha_i^k$或$\beta_i^k$，随后为每个lower分别生成衰减率不超过upper的参数约束。对应cell约束直接构造
$$
p_{\mathrm{upper}}(u)-\sum_j p_{\mathrm{lower}_j}(u)\ge0
\quad\text{for all }u\in[0,1]^{d_C},
$$
其中$d_C$只包含该cell的连续维。空lower列表按零和处理，生成$0\le upper$的cell区域约束。

`distributions/polynomial_bgd.py`中的`symbolic_polynomial_bgd_template(shape, degree, context, name_prefix=...)`以已有BGD的几何框架和网格为形状，先执行不跳过静态零Dirac的结构标准化，再创建统一的PolynomialBGD。为兼容旧调用，`degree`为整数$d$时仍枚举活动连续维上总次数不超过$d$的全部单项式：
$$
\{u^e:\sum_i e_i\le d,\ e_i=0\text{ if dimension }i\text{ is Dirac}\}.
$$
当`degree`为长度等于状态维数的向量$\vec d$时，枚举逐维次数受限的张量积基：
$$
\{u^e:0\le e_i\le d_i,\ e_i=0\text{ if dimension }i\text{ is Dirac}\}.
$$
循环语义使用后一种形式，从而允许不同程序变量采用不同次数。Dirac条件是逐cell
判断的：即使某变量的全局模板次数$d_i>0$，在该变量退化为点质量的cell中仍只生成
$e_i=0$的项。

每个单项式系数以及每个$\alpha_i,\beta_i$都是在`ConstraintContext`中按稳定名称声明的独立`ParameterVariable`。构造器同时把该模板的全部衰减区间约束和cell区域非负约束加入context；它只返回符号BGD并更新精确约束IR，不调用Adapter或求解器。重复使用同一`name_prefix`会因参数重名而被拒绝，避免两个模板意外共享未知量。

第四阶段测试继续位于`tests/test_polynomial_bgd.py`，覆盖衰减区间和区域非负约束、PolynomialBGD比较、不同中心/网格/周期对齐、对齐后的衰减幂、多个lower的精确多项式求和、空和、禁止求解器工厂、总次数多元基、Dirac维单项式消除、参数稳定命名及`ConstraintProblem`声明闭包。

第五阶段在`PolynomialCellOps.convolve_uniform_dim`中实现单个cell沿指定维与均匀噪声的精确卷积。设连续源区间为$[a,b]$、$s=b-a>0$，噪声区间为$[L,H]$、$w=H-L$，目标分段为$[c,d]$，目标局部坐标为$v\in[0,1]$。若$F$是源局部多项式$p$关于卷积维局部坐标的原函数，则目标cell多项式为
$$
q(v)
=
\frac{s}{w}
\left[
F\left(
\min\left(1,\frac{c+(d-c)v-L-a}{s}\right)
\right)
-
F\left(
\max\left(0,\frac{c+(d-c)v-H-a}{s}\right)
\right)
\right].
$$
目标轴用$a+L,a+H,b+L,b+H$的有序去重集合分段，因此每个目标cell内的`min`和`max`分支固定；实现只需调用`antiderivative`和`affine_substitute`，并保留其余状态变量及参数系数。因子$s/w$同时包含物理坐标积分的Jacobian和均匀噪声密度。若源轴是Dirac区间$[a,a]$，源多项式不依赖该状态维，卷积结果是在$[a+L,a+H]$上的$p/w$，该维由Dirac变为连续。

`PolynomialMUD.convolve_uniform(dim, low, high)`合并所有源cell产生的四类断点，并在相同目标cell上通过`PolynomialCellOps.add`累加贡献。卷积维次数至多增加1，其他维次数不变；参数多项式系数不会被求值或近似。空MUD仍返回`PolynomialMUD` family的规范空分布。该操作不归一化输入，但由于均匀噪声质量为1，精确保持原MUD总质量。

`BGD.convolve_uniform`现在按cell family分派。PolynomialBGD沿用原BGD的目标框架与有限相交块枚举：目标中心为$[A_i+L,B_i+H]$，左右周期和衰减率不变；每个源尾块先乘对应有限次$\alpha_i^k$或$\beta_i^k$，转换到全局坐标并做PolynomialMUD精确卷积，再截断到目标块、转回目标局部坐标并精确相加。Uniform具有有限支撑，所以对任一目标基础块只有有限多个源周期产生正长度交集。PolynomialBGD路径不接受`max_fn`、`bound_factory`或`max_interval`，因为这些参数只用于MassMUD上近似，精确多项式路径不需要新上界变量。

第五阶段测试分别位于`tests/test_polynomial_mud.py`和`tests/test_polynomial_bgd.py`，覆盖常数三角形卷积、线性到二次的次数增长、非单位源区间与噪声宽度、多个源cell在共享目标段求和、Dirac到均匀密度、符号参数系数、多维单轴卷积、中心框架移动、符号衰减尾块的有限求和、宽噪声跨多个周期以及MUD/BGD质量保持。此外使用80组有理区间和三次多项式，在每个输出分段的三个有理点将生成结果与直接物理积分交叉比较，全部精确相等。

第六阶段的精确先验构造位于`preprocessing/polynomial_prior_prep.py`。`Uniform[a,b]`直接构造成密度$1/(b-a)$的单cell PolynomialBGD；有限`Mapping`构造成带零密度间隙的Dirac cells；数值先验构造成质量1的Dirac cell。多个先验通过`independent_product`形成联合分布。Normal和Exponential没有有限精确分段多项式表示，因此当前精确模式显式拒绝它们，不会隐式转成MassMUD上界。

多项式AST语义位于`semantics/program.py`，公共结果`PolynomialProgramResult`包含未归一化PolynomialBGD、变量维度顺序、`ConstraintProblem`和参数多项式质量目标。具体先验和无循环语句不引入未知参数，因此其约束问题仍为空；循环语义则在同一个结果类型中返回模板参数、Park约束和几何尾质量辅助参数。`ProgramStructure.build_polynomial_semantics()`只构造语义，不直接调用Adapter。

赋值语义包括常数平移、指定变量与Uniform噪声的精确卷积，以及用独立精确分布替换指定维度。状态条件先转成单变量区间并分别截断真假分支，执行分支后精确相加，因此变量间相关性由外层BGD及其多元cell多项式保留。数值`if`和pGCL概率选择使用精确有理权重混合分支。`observe(g)`返回$\mathbf 1_g\mu$，不做后验归一化。当前支持非嵌套`while`的Park语义；无界`loop`、`tick`、嵌套循环及循环体内的独立分布替换仍显式拒绝。

程序语义测试位于`tests/test_polynomial_program_semantics.py`，覆盖Uniform与离散联合先验、Uniform加法与减法卷积、常数平移、分布替换、状态分支相关性、概率选择、数值`if`、observe子概率质量、空约束问题、Park循环约束，以及对非多项式精确先验的边界检查。

### 实验性SCIP后端

SCIP接入位于`solvers/scip.py`，依赖PySCIPOpt 6.2，可在PPL环境中用
`python -m pip install -r requirements-scip.txt`安装。该后端直接消费
`ConstraintProblem`，不会改变或提前数值化语义IR中的精确有理系数。
`ParameterConstraint`被编译为SCIP有界多项式约束；
`PolynomialIdentity`按状态单项式逐系数展开为参数多项式等式。目标函数也可以是
任意`ParameterPolynomial`，通过一个带区间界的辅助变量转成SCIP线性目标，因此
衰减率、cell系数和其他辅助参数可以联合进入同一个非凸全局优化模型。

对区域约束
$$
p(u,\theta)\ge0,\qquad u\in[0,1]^d,
$$
当前后端使用有限次Putinar型证书
$$
p=s_0+\sum_{i=1}^d s_i\,u_i(1-u_i),
$$
其中每个$s_i$是SOS。由于SCIP不原生支持半正定矩阵，后端以
$s=z^\mathsf TLL^\mathsf Tz$参数化SOS，并把多项式恒等展开为逐系数等式。
`certificate_degree`控制证书次数；省略时取不小于目标多项式次数的最小偶数。
所有原参数必须由调用者给出有限上下界，因子$L$也必须通过`factor_bound`有界，
这是SCIP执行有界非凸全局优化所需要的。严格不等式用显式
`strict_epsilon>0`转成非严格数值约束。

该转换对找到的精确恒等证书在数学上是充分条件，但并不完备：固定证书次数或过小
的因子界可能使真实非负多项式无法通过。实际求解又使用浮点容差，所以当前返回的是
数值证书候选及其最大系数等式残差，不是经过有理重构或区间算术复核的形式化证明。
后续若上界结论需要可核验证书，应增加高精度重构和严格残差验证，或改接原生SDP/SOS
求解链。

快速回归测试位于`tests/test_scip_polynomial_solver.py`，同时覆盖直接由
`PolynomialBGD.le_constraints`生成约束再交给SCIP的端到端路径。其中联合优化例
同时包含衰减率$\alpha$、cell系数$c$及区域约束，SCIP得到
$\alpha\approx c\approx0.5$和目标$t\approx0.0625$，与解析解一致。
`benchmarks/scip_polynomial_probe.py`用于观察规模增长。2026-07-28在当前PPL环境、
PySCIPOpt 6.2.1和SCIP 10.0.2下，测试
$$
\min t\quad\text{s.t.}\quad
t-\prod_{i=1}^d u_i(1-u_i)\ge0\quad\forall u\in[0,1]^d
$$
得到如下结果：

| 维数 | 多项式/证书次数 | SOS因子变量 | 系数约束 | 时间上限 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 4 | 4 | 10秒 | 约0.004秒证明最优，$t=0.25$ |
| 2 | 4 | 33 | 16 | 10秒 | 找到$t=0.0625$，但对偶界约0.00080，达到时限 |
| 3 | 6 | 375 | 85 | 60秒 | 找到$t=0.015625$，但对偶界为0，达到时限 |

因此SCIP原型已经证明“衰减率与多项式联合建模”在接口和小规模数值上可行，但也显示
分解式SOS会引入大量双线性对称性：候选解往往很快正确，全局最优性证明从二维开始就
可能很慢。它适合当前阶段做小规模端到端验证和寻找候选证书，不应据此替代后续原生
SDP后端；后续还应比较原生PSD建模，以及Bernstein分块充分条件在保守性和速度之间的
权衡。

### SCIP Adapter框架接入

`Adapter/scip_adapter.py`中的`SCIPAdapter`同时提供两条有意分离的入口：

- 旧框架继续使用`Adapter.Expr`和标量cell的MassBGD。`main.py`现在接受
  `solver.name = "scip"`，随后原有`ProgramStructure.solve_bgd(..., method="Park")`
  可以不修改语义构造代码而调用SCIP。`solve_bgd_expr`以最终BGD质量为最小化目标。
- 新的精确多项式框架调用`SCIPAdapter.solve_polynomial(...)`，把
  `ConstraintProblem`、显式参数界和`ParameterPolynomial`目标委托给
  `SCIPPolynomialSolver`。这条路径保留区域多项式和SOS证书，不经过旧`Expr`。

旧`Expr`路径支持加、减、乘、除和整数次幂。由于SCIP全局优化需要有限定义域，普通
模板变量默认使用可配置的`[variable_lower_bound, variable_upper_bound]`，衰减率
变量和Diabolo系数`c_w*`使用`[0,1-strict_epsilon]`；`variable_bounds`可按名字覆盖。
区间传播同时为非线性目标和辅助表达式计算有限界。若除数区间包含0或幂不是整数，
adapter会显式拒绝，而不是生成定义域不清楚的SCIP表达式。

旧框架中的`Max(a,b)`使用一个连续辅助变量、一个二进制变量和四个由传播区间确定的
Big-M约束精确编码，因此不是只有$r\ge a,b$的上包络松弛。代价是包含符号max的模型
成为混合整数非线性问题；变量界过宽会削弱其全局界。`!=`目前不支持。

SCIP达到时限或相对gap限制时，只要存在数值可行解，adapter默认返回当前最好解，因为
它仍可作为候选BGD上界；`last_stats`记录status、对偶界、gap、节点数、求解时间和独立
复算的最大约束违反量。若调用场景必须取得SCIP的`optimal`状态，应设置
`require_optimal=true`，此时`gaplimit`和`timelimit`都会抛出错误。无论何种设置，
浮点可行性仍不等价于形式化证明。

`bgd_config.json`已经加入完整`solver.scip`配置示例。框架接入测试位于
`tests/test_scip_adapter.py`，覆盖一般非线性目标、精确max、旧
`build_bgd_leq/solve_bgd_expr`生命周期、新多项式IR委托、主配置注册，以及真实
`ProgramStructure`概率Park循环。测试循环
`while(1/2) { x := x - 1 }`生成7个模板变量和26个SCIP约束，在当前环境约0.003秒内
得到质量约1、左衰减率约0.5的结果。

### PolynomialBGD Park循环与add_uniform验证

`semantics/program.py`现在对非嵌套`while`构造Park前不动点。对循环入口$\mu_0$、
守卫$g$、循环体抽象变换$T$和符号模板$I$，直接调用
`leq_sum([mu_0, T(1_g I)], I)`生成
$$
\mu_0+T(\mathbf1_g I)\le I,
$$
而不会先用符号max构造两个BGD之和。循环的输出仍为
$\mathbf1_{\neg g}I$。模板的所有衰减率、cell多项式系数、非负约束和比较约束均保留
在同一个`ConstraintContext`中。

循环模板中心域包含入口支撑、单步赋值支撑和有限守卫边界。常数平移量决定对应方向的
尾周期；Uniform卷积宽度决定尾周期。有限次具体展开只用于收集中心内的普通断点和
Dirac断点，不产生约束，也不把不断增长的展开支撑全部塞入中心域。

模板次数默认由同一组有限具体展开逐变量推理。对变量顺序
$(x_1,\ldots,x_n)$，在循环入口seed和所有probe的全部PolynomialBGD cell中取
$$
d_i=\max_C\deg_{u_i}p_C.
$$
零多项式的次数按$-1$处理，但每维最终次数至少为0。所得向量直接作为逐维模板次数，
不会再压成统一总次数。`template.polynomial_loop_degree`默认为`"infer"`；也可以显式
给出整数（复制到每个变量）或长度为$n$的整数数组。配置
`template.polynomial_loop_degree_increment`默认为0，会在构造模板前给每个$d_i$统一
增加该非负整数。`PolynomialProgramResult.loop_template_degrees`记录每个循环实际采用
的次数向量，`main.py`也会打印它。

这里的“推理”是有限probe上的语法/语义启发式，不是循环所有迭代上的次数不动点；
例如重复Uniform卷积的真实有限展开次数会继续增长。Park约束仍保留模板与循环体结果
之差的精确高次多项式，因此次数选择只限制待求上界模板的表达能力，不截断或近似循环体
卷积语义。

符号衰减率使BGD质量包含$1/(1-\alpha_i)$或$1/(1-\beta_i)$。函数
`polynomial_bgd_mass`逐尾方向使用`ConstraintContext.exact_positive_quotient`，
以新参数$q$和恒等式
$$
(1-\gamma)q=m
$$
精确提升每个几何尾质量。因此传给SCIP的目标仍是ParameterPolynomial，不会在语义IR
中引入有理函数。`default_polynomial_variable_bounds`为衰减率、cell系数和质量辅助变量
生成有限求解界；这些界属于求解配置，不属于精确语义。

默认`main.py`现在读取`benchmarks/PLDI22/add_uniform.txt`，使用PolynomialBGD语义和
SCIP，不再默认走Rust/MassBGD路径。变量顺序为`(y, x)`，两次probe推理出模板次数
`(0, 1)`：离散计数变量`y`不需要连续幂，而Uniform累加变量`x`使用一次模板。该模板
产生47个参数变量和90个语义约束；SOS编译后有340个因子变量和279个SCIP约束。在当前
环境的60秒限制内能找到数值可行候选，但未证明全局最优，退出分布质量上界约
$$
2.0532836257.
$$
该benchmark的真实退出质量为1，所以结果仍较松，但明显好于统一零次模板的
$2.6547036177$。将每维统一升一次得到`(1, 2)`，模型增长到95个语义参数、1156个SOS
因子变量和582个SCIP约束；60秒内没有找到可行证书。因此当前默认使用推理次数，
升次只保留为显式实验开关，不做失败后的自动回退。

`benchmarks/validate_add_uniform_polynomial.py`独立使用负二项分布与Irwin-Hall密度计算
真实退出密度。在$x\in[0,8]$的4000个区间中点上，求解所得PolynomialBGD均不低于真实
密度；本次`(0,1)`模板的最小采样余量约0.03333，出现在$x=7.999$。该采样只作为独立
数值交叉检查；区域上界的正式求解条件仍来自完整Park和SOS约束，而不是采样点。

### 其他PLDI22 benchmark实验

2026-08-03复查发现，最初对`exfig6`和`cavex5`得到的`infeasible`不是可靠的求解能力
结论，而是循环模板遗漏周期Dirac结构导致的语义构造bug。旧MassBGD路径会调用
`_add_probe_diracs_to_template`，把probe中的全局Dirac点按左右尾周期映射到边缘block；
最初的PolynomialBGD路径只把这些点加入中心block。于是离散计数变量在中心为Dirac，
尾部却只有连续cell。

最小反例为
```text
prior: x := 0
while (1/2) { x := x + 1 }
```
错误模板的中心含$x=0,1$两个Dirac，但右尾只有连续区间。Park约束依次要求
$$
I(0)\ge1,\qquad I(1)\ge\tfrac12I(0),\qquad
0\ge\tfrac12I(1).
$$
最后一个0来自“连续上界在Dirac子区间上的质量为0”，因此模型必然矛盾。这不是
PolynomialCell的密度/质量归一化错误；连续测度确实不能支配正的点质量。错误在于模板
本应在右尾提供周期Dirac cell，却没有构造出来。

`semantics/program.py`现在收集seed和probe的全局Dirac点，按中心边界和左右周期映射到
每个方向block的局部坐标，再执行BGD标准化。上述最小反例的右尾断点由`(0,1)`变为
`(0,1,1)`，SCIP随后约0.01秒证明最优并得到质量约1。该回归同时由语义结构测试和
SCIP端到端测试覆盖。`ProgramStructure`和主配置缺省的`loop_unroll_iterations`统一为2，
以保证单步平移确定中心边界后，至少还有一次probe可暴露落入周期尾部的Dirac相位。

修复后的实验如下。参数系数、尾质量辅助量和SOS因子界均为20，可行性容差为
$10^{-7}$。表中的长实验都关闭了SCIP通用对称性处理，避免分解式SOS的列置换和符号
对称在presolve中生成大量额外约束：

| benchmark | 模板次数 | 语义参数 | 总语义约束 | SOS因子 | SCIP约束 | 时间与结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `exfig6` | 零次`(0,0)` | 143 | 406 | 1252 | 1019 | 60秒内找到质量13.031858的候选，残差约$9.7\times10^{-13}$ |
| `exfig6` | 推理`(2,0)` | 395 | 406 | 4696 | 2567 | 标准模式600秒无候选；可行性模式300秒找到质量222.426772、残差约$5.7\times10^{-14}$的候选；再以完整SOS因子热启动优化600秒，目标未改善，对偶界0.029195 |
| `cavex5` | 零次`(0,0)` | 513 | 1327 | 2331 | 2044 | 可行性模式300秒无候选 |
| `cavex5` | 推理`(1,0)` | 1013 | 1327 | 7325 | 5523 | 设置600秒，在根节点内部实际运行1171秒，找到质量9.650010、残差约$9.99\times10^{-8}$的候选，对偶界0.517628 |
| `cavex7` | 推理 | - | - | - | - | 循环体内`u := Uniform(0,1)`分布替换尚不支持 |
| `geo` | 推理 | - | - | - | - | 同样因循环体内分布替换被拒绝 |

SCIP最初在`cavex5`的7325个SOS因子上识别出653个对称生成元，并在进入根节点前添加
29350条对称约束；30秒几乎全部耗在presolve。`SCIPPolynomialSolver`现在显式设置
`misc/usesymmetry=0`。关闭后，同样30秒能够进入根节点并推进对偶界。这只移除了因子分解
表示引入的搜索冗余，不改变SOS多项式恒等式。该策略通过
`solver.scip.use_symmetry=false`显式启用；默认值仍为`true`，因为小模型（例如
`add_uniform`）保留SCIP对称性处理时在短时限内通常能找到更好的候选。

`solver.scip.feasibility_emphasis`可以启用SCIP的可行性强调和激进启发式。求解结果还会
返回全部`factor_values`，并可通过`initial_values`把参数和SOS因子完整送入下一阶段；
`exfig6`的两阶段实验使用了该接口。可行性强调有助于区分“模型无解”和“没有找到
incumbent”，但它找到的第一个证书可能很松，本身不能体现升次模板的质量优势。

当前`time_limit`仍不是严格墙钟上限。SOS因子和逐系数恒等式在`model.optimize()`之前
生成，不计入SCIP时限；此外SCIP在一次根节点LP、分离或启发式调用内部也可能晚于限制
返回，所以`cavex5`设置600秒却实际运行约1171秒。后续需要增加编译阶段规模预估，并用
外层进程超时实现严格的端到端限制。

因此目前可以区分两个问题：周期Dirac遗漏是已经确认并修复的正确性bug；修复后
`exfig6`零次模板与旧分段常数方案一样恢复可行。推理高次模板也已在`exfig6`和
`cavex5`上找到数值可行证书，不应再称为不可解；但分解式SOS的规模和非凸性使优化
质量很差。`exfig6`高次候选222.43远差于零次候选13.03，`cavex5`又没有在相同后端和
预算内得到零次候选，所以这些实验尚不能证明多项式模板带来了更紧的上界。下一阶段的
重点应是原生PSD/SOS后端或更好的凸化，而不是继续单纯增加SCIP时限。
