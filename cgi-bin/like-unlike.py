#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import cgitb
cgitb.enable()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

import embutils
from kpdemosdev import *

populate_js = '''
function populateTextField() {
document.getElementById("inputted_text").value = "auto\\nauto like bussi\\nauto unlike bussi\\ncluster auto\\nbussi like juna\\nhiiri like näppäimistö\\nhiiri unlike näppäimistö";
}
'''

column_names = ["Query", "Result"]

def make_cluster(clusterword, vecs):
    plt.clf()
    similarityfactor = 1.0
    nwords = 25
    likes = []
    threshold = 0.4
    get_words = lambda x: list(map(first, x))
    words_with_weights = vecs.like(clusterword, nwords)
    words = get_words(words_with_weights)

    groups = {}
    groups_with_weights = {}
    associations = {}

    groups[clusterword] = words
    groups_with_weights[clusterword] = words_with_weights
    for word2 in words:
        if word2 != clusterword:
            clusterwords_with_weights = vecs.like(clusterword, word2, nwords, similarityfactor)
            groups[word2] = get_words(clusterwords_with_weights)
            groups_with_weights[word2] = clusterwords_with_weights

    for key in groups:
        associations[key] = {}
        for key2 in groups:
            if key == key2:
                continue
            associations[key][key2] = len(list(filter(lambda x: x in groups[key2], groups[key])))

    def clusters_with(a, b):
        return float(associations[a][b])/nwords > threshold

    G=nx.Graph()
    
    for a in groups.keys():
        for b in groups.keys():
            if a == b:
                continue
            if clusters_with(a, b):
                G.add_edge(a, b, weight=vecs.get_distance(a, b))

    edges = [(u,v) for (u,v,d) in G.edges(data=True)]
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos,node_size=60)
    nx.draw_networkx_edges(G,pos,edgelist=edges,
                           width=1,alpha=0.1,edge_color='b')

    def add_to_verticals(posdict, delta = 0.05):
        retval = {}
        for item, key in posdict.items():
            retval[item] = (key[0], key[1] + delta)
        return retval

    nx.draw_networkx_labels(G, add_to_verticals(pos), font_size=10, font_family='sans-serif')

    plt.axis('off')
    def onlyascii(char):
        if ord(char) < 48 or ord(char) > 127:
            return ''
        else:
            return char
    asciiclusterword = ''.join(filter(onlyascii, clusterword))
    writefile = wrkdir + "/tmp/like_cluster_for_" + asciiclusterword + ".png"
    linkaddress = 'https://kielipankki.fi/tools/demo/kielipankki-tools/tmp/like_cluster_for_' + asciiclusterword + ".png"
    plt.savefig(writefile, dpi = 120) # save as png
    return [['', '<img src="' + linkaddress + '" />']]

def process_input(text, vecs):
    nwords = 15
    def process_result(op):
        return (list(map(lambda x: ['', x], map(first, op))))
    retval = []
    for line in text.split('\n'):
        line = line.strip()
        parts = line.split(' ')
        retval.append([line, ''])
        if len(parts) == 1:
            retval += process_result(vecs.like(parts[0], nwords))
        elif len(parts) == 2 and parts[0] == "cluster":
            retval += make_cluster(parts[1], vecs)
        elif len(parts) == 3 and parts[1] in ('like', 'unlike'):
            if parts[1] == 'like':
                retval += process_result(vecs.like(parts[0], parts[2], nwords))
            else:
                retval += process_result(vecs.unlike(parts[0], parts[2], nwords))
        else:
            retval += ['', 'parsing error']
    return retval

def print_content():
    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    if "input" in form:
        inputval = form["input"].value
    if "file" in form and form["file"].filename != "":
        inputval = text_from_file(form["file"]).decode("utf-8")
    if inputval != "":
        try:
            vecfile = "/srv/vecs/all_vec.bin"
            if "vec_choice" in form:
                if form["vec_choice"].value == 'ftc_vecs':
                    vecfile = "/srv/vecs/all_vec.bin"
                elif form["vec_choice"].value == 's24_vecs':
                    vecfile = "/srv/vecs/s24_vec.bin"
            vecs = embutils.WordEmbeddings()
            vecs.load_from_file(vecfile)
            out = process_input(str(inputval), vecs)
            session_key = hashlib.md5(inputval.encode("utf-8")).hexdigest()
#            write_excel(column_names + out, session_key, "Output from like-unlike")
#            <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="like_unlike.xlsx" role="button">Download Excel spreadsheet</a>
            write_txt(rows2tsv(out), session_key)
            result += "<p>Result:</p>\n"
            result += '''
            <div class="row">
            <div class="col-md-auto">
            <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.txt" download="like_unlike.txt" role="button">Download text</a>
            </div>
            </div>
            '''.format(html_root = hostname, filename = session_key)
            result += make_table(out, header = column_names) + "\n"
        except Exception as ex:
            result += str(ex)

    body = wrap_in_tags("like-unlike demo", "h2")
    body += '''
<h6>Query word embeddings.</h6>
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">
        Enter queries, one per line, into the text box. Possible queries are:
        <ul>
          <li>A single word, returning a list of neighbour words</li>
          <li><code>word1 like word2</code>, returning a list of words similar to <code>word1</code> <i>in the sense of</i> <code>word2</code>, and vice versa. For example, <code>mouse like keyboard</code></li> is different from <code>mouse like hamster</code>.
          <li><code>word1 unlike word2</code>, returning a list of words similar to <code>word1</code> <i>but not in the sense of</i> <code>word2</code>. For example, distributionally, "happy" is similar to "sad", but <code>happy unlike sad</code> tries to push away the words more like "sad".</li>
          <li><code>cluster word</code>, which draws a graph of the relationships around <code>word</code> in the sense of <code>like</code></li>
        </ul>
      </p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">
      </p>
    </div>
  </div>
</div>
<form method="post" action="/cgi-bin/{scriptname}" enctype="multipart/form-data">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto">
        <textarea name="input" type="text" rows="10" cols="60" class="form-control" id="inputted_text" placeholder="Input words go here."></textarea>
      </div>
      <div class="row-md-auto">
        <div class="radio"><label><input type="radio" name="vec_choice" value="ftc_vecs" checked="checked">Word embeddings from newspaper corpus</label></div>
        <div class="radio"><label><input type="radio" name="vec_choice" value="s24_vecs">Word embeddings from web forum corpus</label></div>
      </div>
    </div>
    <div class="row">
      <div class="col-md-auto">
        <button type="button" class="btn btn-default" onclick="populateTextField()"><b>Or</b> populate with demo text</button>
      </div>

      <div class="col-md-auto"><p><b>Or</b></p></div>
      <div class="col-md-auto pl-0">
        <input type="file" class="form-control-file" name="file" id="submit_file">
      </div>
    </div>
    <div class="row">
      <div class="col-md-auto">
        <button type="submit" class="btn btn-primary" id="submit_button">Submit</button>
      </div>
    </div>

  </div>
</form>
<div class="row">
  <div class="col-md-auto">
    {content}
  </div>
</div>
<p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
'''.format(scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)

    sys.stdout.buffer.write(wrap_html(make_head(title = 'like-unlike demo', scripts = (populate_js,)), wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    print_content()
