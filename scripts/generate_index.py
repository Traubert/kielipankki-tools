import sys
from jinja2 import Template

template = Template(open(sys.argv[1]).read())

categories = [
    { "title": "Rakenteellinen analyysi",
      "items": [
          { "title": "fintag",
            "script": "fintag.py",
            "description": "Tekstin perusmuotoistus, taivutuksen ja nimettyjen ilmausten tunnistus"
          },
          { "title": "finparse",
            "script": "finparse.py",
            "description": "Tekstin morfosyntaktinen dependenssijäsennys"
          },
          { "title": "ofitwol",
            "script": "ofitwol.py",
            "description": "Morfofoneeminen analyysi ofitwol-kaksitasosäännöillä"
          },
      ]
    },
    { "title": "Puhuttu kieli",
      "items": [
          { "title": "Puheentunnistus",
            "script": "asr.py",
            "description": "Suomen kielen puheentunnistus (Aalto-ASR & Kaldi)"
          },
          { "title": "finnish-forced-align",
            "script": "forced-align.py",
            "description": "Puheen kohdistus tekstiin"
          },
      ]
    },
    { "title": "Tekstianalyysi",
      "items": [
          { "title": "FinnSentiment",
            "script": "predict/finsent.py",
            "description": "Tekstin sentimenttianalyysi"
          },
          { "title": "lemmarank",
            "script": "lemmarank.py",
            "description": "Tekstille leimallisten sanojen ja nimettyjen ilmausten löytäminen ja vertailu"
          },
      ]
    },
    { "title": "Sanasemantiikka",
      "items": [
          { "title": "like-unlike",
            "script": "like-unlike.py",
            "description": "Vastaavuuksia (<i>like</i>) ja vastakohtaisuuksia (<i>unlike</i>) tukeva haku sanaupotusaineistoista"
          },
          { "title": "lemmamatch",
            "script": "lemmamatch.py",
            "description": "Tekstihaku perusmuotojen ja sanasemantiikan avulla",
            "extra": '''
<small>
<ul>
  <li><a href="cgi-bin/lemmamatch-termipankki.py">Tieteen termipankkin sanalistoihin perustuva esimerkki</a></li>
  <li><a href="cgi-bin/lemmamatch-wordvecs.py">Sanasemantiikkaesimerkki</a></li>
  <li><a href="cgi-bin/lemmamatch-rules.py">Esimerkki sääntöjen kirjoittamisesta</a></li>
  </ul>
</small>                                                                                                               
'''
          },
          { "title": "FinnWordNet",
            "script": "fiwn.cgi",
            "description": "Tietokantahaku semanttisesta sanakirjasta"
          },

      ]
    },
    { "title": "Apuvälineitä",
      "items": [
          { "title": "Monikielinen tekstintunnistus",
            "script": "ocr.py",
            "description": "Eristä teksti kuvatiedostoista (jpg, png, pdf, ...)"
          },
          { "title": "fintok",
            "script": "fintok.py",
            "description": "Tekstin tokenisointi"
          },
          { "title": "XML-strip",
            "script": "xml-strip.py",
            "description": "Juoksevan tekstin eristäminen XML-dokumentista jatkokäsittelyä varten"
          },
          { "title": "Latauspalvelu",
            "script": "../download/index.html",
            "description": "Kehitysvaiheessa olevien aineistojen (sanavektorit) latauspalvelu"
          },

      ]
    },
    

]

print(template.render(categories = categories))
