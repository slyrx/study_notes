## 将图片嵌入文档中
方法是使用base64编码。
步骤： 
1. 将图片或截图保存在本地； 
2. 使用在线工具将图片转码至base64编码；("http://imgbase64.duoshitong.com/", "https://tool.css-js.com/base64.html")； 
3. 在文档中插入编码：

```
![image][data:image/png;base64, ......]
```

base64编码一般很长，直接将编码放入段落内部会影响正常编辑。
通常是将base64编码定义一个中间变量，将变量放到文档末尾。
```
![image][tmp]
your document here ...

[tmp]:data:image/png;base64, ......
```
使用该技巧的时候需要注意，并不是所有的Markdown编辑器都支持这种方法。
一些Markdown编辑器只支持特定的图片格式。如有道云笔记只支持png格式的图片编码。
需要在保存图片文件的时候加以注意。
