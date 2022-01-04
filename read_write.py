import codecs

file = codecs.open("html/bitcoin1.html", 'r', "utf-8")
template=file.read().replace("%s", "50000")
f = open('html/bitcoin1.html', 'w')
f.write(template)
f.close()