from html import escape
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


OUT = Path("Ecommerce_Recommender_System_Publishable_Paper.docx")


def p(text="", style=None, align=None):
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    runs = []
    for line_index, line in enumerate(str(text).split("\n")):
        if line_index:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(f"<w:r><w:t xml:space=\"preserve\">{escape(line)}</w:t></w:r>")
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def heading(text, level=1):
    return p(text, style=f"Heading{level}")


def bullet(text):
    return (
        "<w:p><w:pPr><w:pStyle w:val=\"ListParagraph\"/>"
        "<w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr></w:pPr>"
        f"<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"
    )


def table(headers, rows):
    cols = len(headers)
    grid = "".join('<w:gridCol w:w="2400"/>' for _ in range(cols))
    def cell(text, bold=False):
        b = "<w:b/>" if bold else ""
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r><w:rPr>{b}</w:rPr><w:t>{escape(str(text))}</w:t></w:r></w:p></w:tc>"
        )

    header_xml = "<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>"
    row_xml = "".join("<w:tr>" + "".join(cell(value) for value in row) + "</w:tr>" for row in rows)
    return (
        "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders><w:top w:val=\"single\" w:sz=\"4\"/><w:left w:val=\"single\" w:sz=\"4\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\"/><w:right w:val=\"single\" w:sz=\"4\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\"/><w:insideV w:val=\"single\" w:sz=\"4\"/></w:tblBorders>"
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{header_xml}{row_xml}</w:tbl>"
    )


def section_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


body = []

body.extend([
    p("A Hybrid Recommendation System for E-Commerce Product Discovery", style="Title", align="center"),
    p("Research Paper Prepared from the Ecommerce Recommender System Project", align="center"),
    p("Author: Your Name", align="center"),
    p("Affiliation: Your Institution / Department", align="center"),
    p("Date: June 2026", align="center"),
    p(),
    heading("Abstract", 1),
    p(
        "E-commerce platforms require recommendation systems that can guide users toward relevant "
        "products while handling sparse interaction histories and diverse catalog content. This paper "
        "presents a Flask-based hybrid recommender system that combines collaborative filtering, "
        "content-based filtering, product popularity, and adaptive rank fusion. The implementation uses "
        "Amazon product data containing product identifiers, user identifiers, ratings, categories, "
        "product descriptions, prices, and image links. Content-based recommendations are generated using "
        "TF-IDF features and cosine similarity, while collaborative filtering uses item-item and user-user "
        "nearest-neighbor scoring over a user-item rating matrix. Experimental evaluation compares "
        "collaborative filtering, content-based filtering, and the hybrid approach using Precision@K, "
        "Recall@K, F1@K, and NDCG@K at K=5 and K=10. Results show that the content-based model performs "
        "strongly in sparse settings, while the hybrid model provides a balanced recommendation strategy "
        "by combining user behavior, item similarity, profile content, and popularity signals."
    ),
    p("Keywords: recommender system, e-commerce, hybrid filtering, collaborative filtering, content-based filtering, TF-IDF, Flask"),
    section_break(),
    heading("1. Introduction", 1),
    p(
        "The rapid growth of online shopping has created a product discovery challenge for customers and "
        "retail platforms. A large catalog improves choice, but it also increases the difficulty of finding "
        "products that match individual preferences. Recommendation systems address this problem by ranking "
        "products according to estimated user interest. In e-commerce, such systems can improve navigation, "
        "reduce search effort, and increase the likelihood of meaningful product engagement."
    ),
    p(
        "This project implements an e-commerce recommendation system with a web interface and an evaluation "
        "module. The main objective is to combine the strengths of collaborative filtering and content-based "
        "filtering. Collaborative filtering can learn from patterns in user ratings, while content-based "
        "filtering can recommend products with similar names, categories, brands, and descriptions. Because "
        "real product datasets often contain sparse user interactions, the proposed hybrid system also uses "
        "fallback popularity and adaptive weighting."
    ),
    heading("2. Objectives", 1),
    bullet("Develop a product recommendation system for an e-commerce catalog."),
    bullet("Use product metadata to compute content similarity through TF-IDF and cosine similarity."),
    bullet("Use user-item rating interactions to implement collaborative filtering."),
    bullet("Combine content-based, collaborative, profile, and popularity signals in a hybrid ranking model."),
    bullet("Provide recommendations through a Flask web application with login-protected pages."),
    bullet("Evaluate CF, CB, and hybrid approaches using ranking metrics at K=5 and K=10."),
    heading("3. Related Work", 1),
    p(
        "Recommendation systems are commonly divided into collaborative filtering, content-based filtering, "
        "and hybrid methods. Collaborative filtering estimates user preferences from historical interactions "
        "between users and items. Memory-based variants use similarity among users or items, while model-based "
        "variants learn latent representations. Content-based filtering represents products using descriptive "
        "features and recommends items similar to those a user has already selected or rated. Hybrid systems "
        "combine multiple recommendation signals to reduce limitations such as cold start, sparsity, and "
        "overspecialization."
    ),
    heading("4. System Architecture", 1),
    p(
        "The system is implemented as a Python Flask application. The application includes authentication, "
        "a dashboard, a recommendation form, visual analytics, and recommendation result pages. The main "
        "recommendation logic is implemented in app/recommender.py. The web route receives a user identifier "
        "and product identifier, calls the hybrid recommendation function, and renders product names, ratings, "
        "discounted prices, and product images."
    ),
    table(
        ["Component", "Role"],
        [
            ["Flask application", "Serves dashboard, recommendation, visualization, and network pages."],
            ["Data preprocessing", "Loads Amazon product data and cleans missing or invalid values."],
            ["Content-based module", "Builds TF-IDF features from product text and computes cosine similarity."],
            ["Collaborative module", "Builds user-item matrix and computes user-user and item-item similarity."],
            ["Hybrid ranker", "Combines CB, CF, profile, and popularity scores with adaptive weighting."],
            ["Evaluation module", "Reports Precision@K, Recall@K, F1@K, and NDCG@K."],
        ],
    ),
    heading("5. Dataset and Preprocessing", 1),
    p(
        "The project uses an Amazon product dataset stored at data/raw/amazon.csv. The dataset is approximately "
        "4.75 MB and includes product_id, product_name, category, about_product, rating, discounted_price, "
        "img_link, and user_id fields. Records with missing core fields are removed. Ratings are converted to "
        "numeric values, product identifiers are normalized as strings, and multi-value user identifiers are "
        "split and exploded into individual user-product interactions. Duplicate user-product pairs are removed."
    ),
    p(
        "For content modeling, product name, category, extracted brand, and product description are cleaned by "
        "lowercasing text, removing URLs, removing non-alphanumeric characters, and normalizing whitespace. The "
        "final content representation gives additional weight to product name, category, and brand, followed by "
        "the product description."
    ),
    heading("6. Methodology", 1),
    heading("6.1 Content-Based Filtering", 2),
    p(
        "The content-based component represents each product with TF-IDF features. The vectorizer uses English "
        "stop-word removal, unigram and bigram features, sublinear term frequency, Unicode accent stripping, "
        "a maximum document frequency of 0.85, and up to 12,000 features. Cosine similarity is then calculated "
        "between product vectors. Given a target product, the system ranks catalog items by content similarity "
        "and excludes the input item from the results."
    ),
    heading("6.2 Collaborative Filtering", 2),
    p(
        "The collaborative filtering component creates a user-item rating matrix from cleaned interactions. "
        "User-user similarity is computed from mean-centered user rating vectors, and item-item similarity is "
        "computed from mean-centered item rating vectors. For a user with rating history, the system produces "
        "item-neighborhood scores and user-neighborhood scores. These scores are normalized and combined with "
        "a heavier item-neighborhood weight. When collaborative evidence is insufficient, popularity is used "
        "as a fallback signal."
    ),
    heading("6.3 Hybrid Recommendation", 2),
    p(
        "The hybrid model combines content similarity, collaborative filtering, user profile content similarity, "
        "and popularity. The default weighting emphasizes content similarity with cb_weight=0.70 and cf_weight=0.30. "
        "The collaborative contribution is adjusted by a confidence value based on the number of nonzero CF "
        "candidates and the activity level of the user. If CF evidence is weak, more of the collaborative budget "
        "is shifted to profile-based content scores. A small popularity weight is included to stabilize rankings. "
        "The system also applies reciprocal-rank fusion nudges to help strong candidates surface when score scales "
        "are close."
    ),
    heading("7. Experimental Setup", 1),
    p(
        "The evaluation uses a train-test split of the interaction data. The recorded run used 1,169 training "
        "interactions and 293 test interactions, with 983 unique users in training and 271 unique users in testing. "
        "The compared approaches are collaborative filtering (CF), content-based filtering (CB), and the hybrid "
        "method. The metrics are Precision@5, Recall@5, F1@5, NDCG@5, Precision@10, Recall@10, F1@10, and NDCG@10."
    ),
    heading("8. Results", 1),
    table(
        ["Metric", "CF", "CB", "Hybrid"],
        [
            ["Precision@5", "0.0000", "0.1481", "0.1462"],
            ["Recall@5", "0.0000", "0.5185", "0.5192"],
            ["F1@5", "0.0000", "0.2305", "0.2281"],
            ["NDCG@5", "0.0000", "0.6323", "0.4412"],
            ["Precision@10", "0.0000", "0.0727", "0.0864"],
            ["Recall@10", "0.0000", "0.5909", "0.5227"],
            ["F1@10", "0.0000", "0.1295", "0.1482"],
            ["NDCG@10", "0.0000", "0.3894", "0.3433"],
        ],
    ),
    p(
        "The results indicate that collaborative filtering alone performs poorly in this dataset split, likely "
        "because many users have sparse interaction histories. Content-based filtering performs strongly, especially "
        "for NDCG@5 and Recall@10. The hybrid model achieves the best or tied result on three of the eight reported "
        "metrics, including Recall@5 and F1@10. These results support the use of hybrid ranking in sparse e-commerce "
        "settings, while also showing that the content-based signal is the dominant contributor for this dataset."
    ),
    heading("9. Discussion", 1),
    p(
        "The system demonstrates a practical architecture for product recommendation in a small-to-medium e-commerce "
        "dataset. Content metadata is especially valuable when user behavior is limited. The hybrid design improves "
        "robustness by using collaborative information when available and shifting toward profile and popularity "
        "signals when collaborative confidence is low. This adaptive behavior is important for real deployments, "
        "where users may have few prior interactions."
    ),
    p(
        "A limitation of the current evaluation is that the collaborative model is affected by sparsity and may not "
        "receive enough overlapping interactions to produce strong recommendations. Another limitation is that the "
        "project uses text metadata and ratings but does not yet incorporate session behavior, purchase history, "
        "semantic embeddings, or temporal patterns. The metrics are useful for offline comparison, but online A/B "
        "testing would be needed to measure real user satisfaction."
    ),
    heading("10. Conclusion", 1),
    p(
        "This paper presented a hybrid e-commerce recommender system built with Flask, Pandas, NumPy, and scikit-learn. "
        "The system combines TF-IDF content similarity, user-item collaborative filtering, adaptive confidence weighting, "
        "profile-based content scoring, and popularity fallback. Evaluation results show that content-based filtering "
        "is highly effective for the available dataset, while the hybrid model provides a balanced and practical "
        "recommendation strategy. The project is suitable as a foundation for further research into richer hybrid "
        "models, deep learning recommenders, and real-time recommendation APIs."
    ),
    heading("11. Future Work", 1),
    bullet("Improve collaborative filtering with matrix factorization or neural collaborative filtering."),
    bullet("Add semantic text embeddings to capture deeper product meaning beyond TF-IDF terms."),
    bullet("Evaluate with additional metrics such as MAP@K, MRR, coverage, novelty, and diversity."),
    bullet("Introduce real-time event tracking for clicks, cart actions, and purchases."),
    bullet("Deploy the recommender as a REST API and test it with live users."),
    heading("References", 1),
    p("[1] F. Ricci, L. Rokach, and B. Shapira, Recommender Systems Handbook. Springer, 2015."),
    p("[2] J. B. Schafer, J. Konstan, and J. Riedl, \"E-Commerce Recommendation Applications,\" Data Mining and Knowledge Discovery, 2001."),
    p("[3] G. Adomavicius and A. Tuzhilin, \"Toward the Next Generation of Recommender Systems,\" IEEE Transactions on Knowledge and Data Engineering, 2005."),
    p("[4] C. D. Manning, P. Raghavan, and H. Schutze, Introduction to Information Retrieval. Cambridge University Press, 2008."),
    p("[5] scikit-learn Developers, \"scikit-learn: Machine Learning in Python,\" project documentation."),
])


document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
xmlns:w10="urn:schemas-microsoft-com:office:word"
xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
mc:Ignorable="w14 wp14">
<w:body>
{''.join(body)}
<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
</w:body></w:document>'''


styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="320" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="220" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="25"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:qFormat/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''


numbering_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>'''


content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>'''


root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''


doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>'''


with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types)
    zf.writestr("_rels/.rels", root_rels)
    zf.writestr("word/document.xml", document_xml)
    zf.writestr("word/styles.xml", styles_xml)
    zf.writestr("word/numbering.xml", numbering_xml)
    zf.writestr("word/_rels/document.xml.rels", doc_rels)

print(f"Created {OUT.resolve()}")
