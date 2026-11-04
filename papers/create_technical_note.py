"""Render the standalone qwenmem-spine technical note from reviewed content."""
from pathlib import Path
import argparse
import html
import json
import re
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('qwenmem-spine-technical-note.pdf'))
parser.add_argument('--fonts',type=Path,default=Path('/usr/share/fonts/truetype/dejavu'))
args=parser.parse_args()
OUT=args.output
DATA=json.loads(Path(__file__).with_name('technical_note_content.json').read_text())
OUT.parent.mkdir(parents=True,exist_ok=True)
FONTS=args.fonts
for name,file in [('Serif','DejaVuSerif.ttf'),('SerifBold','DejaVuSerif-Bold.ttf'),('SerifItalic','DejaVuSerif-Italic.ttf'),('Sans','DejaVuSans.ttf'),('SansBold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifBold',italic='SerifItalic',boldItalic='SerifBold')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansBold',italic='Sans',boldItalic='SansBold')
INK=colors.HexColor('#17232D');MUTED=colors.HexColor('#56646E');BLUE=colors.HexColor('#235775');RULE=colors.HexColor('#CCD5DB')
WIDTH=A4[0]-108
styles={
 'title':ParagraphStyle('title',fontName='SerifBold',fontSize=20,leading=25,textColor=INK,spaceAfter=5),
 'subtitle':ParagraphStyle('subtitle',fontName='Serif',fontSize=14.4,leading=19,textColor=INK,spaceAfter=11),
 'meta':ParagraphStyle('meta',fontName='Sans',fontSize=8,leading=11,textColor=MUTED,spaceAfter=13),
 'heading':ParagraphStyle('heading',fontName='SansBold',fontSize=10.8,leading=14,textColor=BLUE,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='Serif',fontSize=10.2,leading=13.7,textColor=INK,alignment=TA_JUSTIFY,spaceAfter=8,allowWidows=0,allowOrphans=0),
 'caption':ParagraphStyle('caption',fontName='Sans',fontSize=8.3,leading=11.2,textColor=INK,spaceBefore=2,spaceAfter=6,keepWithNext=True),
 'th':ParagraphStyle('th',fontName='SansBold',fontSize=7.7,leading=10,textColor=INK),
 'td':ParagraphStyle('td',fontName='Sans',fontSize=8.1,leading=10.8,textColor=INK),
 'num':ParagraphStyle('num',fontName='Sans',fontSize=8.1,leading=10.8,textColor=INK,alignment=TA_CENTER),
}
def clean(text):
    for char in ['\u2011','\u2013','\u2014','\u2212']:text=text.replace(char,'-')
    return text

def inline(text):
    value=html.escape(clean(str(text)),quote=True)
    value=re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#235775"><u>{m[1]}</u></link>',value)
    value=re.sub(r'`([^`]+)`',r'<font name="Mono" size="9">\1</font>',value)
    value=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',value)
    return value

def para(text,style='body'):
    return Paragraph(inline(text),styles[style])

def make_table(data,index):
    if index==0:widths=[169,45,48,48,91,WIDTH-401]
    else:widths=[85,49,73,126,82,WIDTH-415]
    cells=[[para(h,'th') for h in data['headers']]]
    cells.extend([[para(cell,'td' if i==0 else 'num') for i,cell in enumerate(row)] for row in data['rows']])
    table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF2F5')),
        ('LINEABOVE',(0,0),(-1,0),.7,BLUE),('LINEBELOW',(0,0),(-1,0),.5,RULE),('LINEBELOW',(0,-1),(-1,-1),.7,BLUE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FA')]),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
    ]))
    return KeepTogether([para(data['caption'],'caption'),table,Spacer(1,9)])

story=[para(DATA['title'],'title'),para(DATA['subtitle'],'subtitle'),para(DATA['meta'],'meta')]
story.append(Paragraph('<b>Abstract.</b> '+inline(DATA['abstract']),styles['body']))
index=0
for section in DATA['sections']:
    story.append(para(section['heading'],'heading'))
    story.extend(para(p) for p in section.get('paragraphs',[]))
    if 'table' in section:
        story.append(make_table(section['table'],index));index+=1
    story.extend(para(p) for p in section.get('paragraphs_after',[]))

class NumberedCanvas(Canvas):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.saved=[]
    def showPage(self):
        self.saved.append(dict(self.__dict__));self._startPage()
    def save(self):
        total=len(self.saved)
        for state in self.saved:
            self.__dict__.update(state)
            self.setStrokeColor(RULE);self.setLineWidth(.4);self.line(54,44,A4[0]-54,44)
            self.setFont('Sans',7.5);self.setFillColor(MUTED)
            self.drawString(54,31,'qwenmem-spine  |  Technical note  |  November 4, 2026')
            self.drawRightString(A4[0]-54,31,f'{self._pageNumber} / {total}')
            super().showPage()
        super().save()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=46,bottomMargin=58,
 title=DATA['title']+': '+DATA['subtitle'],author='',subject='Compiled navigation, verified native Spine integration, November 3 tests and limitations',pageCompression=1)
doc.build(story,canvasmaker=NumberedCanvas)
print(OUT)
