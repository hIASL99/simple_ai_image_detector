"""Branch coverage for aidetect.metadata, built on synthetic files."""
from __future__ import annotations

import io, json, struct, zlib, sys
from pathlib import Path

from PIL import Image, PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

import pytest

from aidetect.metadata import read_metadata

BASE = Image.new("RGB", (32, 24), (90, 120, 160))

def png(**text) -> bytes:
    meta = PngImagePlugin.PngInfo()
    for k, (v, mode) in text.items():
        if mode == "tEXt": meta.add_text(k, v)
        elif mode == "zTXt": meta.add_text(k, v, zip=True)
        elif mode == "iTXt": meta.add_itxt(k, v)
        elif mode == "iTXtz": meta.add_itxt(k, v, zip=True)
    buf = io.BytesIO(); BASE.save(buf, "PNG", pnginfo=meta); return buf.getvalue()

def jpeg(exif=None, xmp=None, quality=92) -> bytes:
    buf = io.BytesIO()
    kw = {}
    if exif is not None: kw["exif"] = exif
    if xmp is not None: kw["xmp"] = xmp if isinstance(xmp, bytes) else xmp.encode()
    BASE.save(buf, "JPEG", quality=quality, **kw)
    return buf.getvalue()

def insert_png_chunk(blob: bytes, ctype: bytes, data: bytes) -> bytes:
    idx = blob.rindex(b"IEND") - 4
    chunk = struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data))
    return blob[:idx] + chunk + blob[idx:]

def insert_jpeg_app11(blob: bytes, jumbf: bytes) -> bytes:
    payload = b"JP" + struct.pack(">H", 1) + struct.pack(">I", 1) + jumbf
    seg = b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload
    return blob[:2] + seg + blob[2:]

SUFFIX = bytes.fromhex("00110010800000aa00389b71")

def box(tag: bytes, label: bytes, body: bytes) -> bytes:
    """A JUMBF superbox: length + 'jumb' + its 'jumd' description + contents."""
    jumd = tag + SUFFIX + b"\x03" + label + b"\x00"
    jumd = struct.pack(">I", len(jumd) + 8) + b"jumd" + jumd
    return struct.pack(">I", len(jumd) + len(body) + 8) + b"jumb" + jumd + body

def jumbf(claim_generator=b"Adobe Firefly 1.2", dst=b"trainedAlgorithmicMedia",
          tag=b"c2pa") -> bytes:
    cbor = (b"\xa2\x6fclaim_generator\x78" + bytes([len(claim_generator)]) + claim_generator
            + b"\x6ac2pa.actions\x81\xa1\x66action\x6bc2pa.created"
            + b"\x78\x38http://c2pa.org/digitalsourcetype/" + dst)
    actions = box(b"cbor", b"c2pa.actions", cbor)
    store = box(b"c2as", b"c2pa.assertions", actions)
    claim = box(b"cacl", b"c2pa.claim", cbor)
    return box(tag, b"c2pa", box(b"c2ma", b"urn:uuid:0f2a-1111", store + claim))

XMP = ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
       '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       '<rdf:Description rdf:about="" {ns}>{body}</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')

def xmp_dst(uri_base, term):
    return XMP.format(ns='xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"',
                      body=f'<Iptc4xmpExt:DigitalSourceType>{uri_base}{term}</Iptc4xmpExt:DigitalSourceType>')

A1111 = ("a cat sitting on a sofa\nNegative prompt: blurry, lowres\n"
         "Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7.0, Seed: 3141592653, "
         "Size: 1024x1024, Model hash: 31e35c80fc, Model: sd_xl_base_1.0")
COMFY = json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 1}},
                    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}}})
COMFY_WF = json.dumps({"nodes": [{"id": 1, "type": "KSampler", "widgets_values": [1, "randomize"]}]})
INVOKE = json.dumps({"app_version": "4.2.9", "generation_mode": "txt2img", "seed": 12, "steps": 30,
                     "model": {"name": "sdxl", "hash": "abc"}, "positive_prompt": "a dog"})
NOVELAI = json.dumps({"steps": 28, "sampler": "k_euler", "seed": 992, "scale": 5.0,
                      "uc": "lowres", "noise_schedule": "native"})

CASES: list[tuple[str, bytes, dict]] = []
def case(name, blob, **expect): CASES.append((name, blob, expect))

# -- 1. PNG text chunk variants
case("A1111 tEXt parameters", png(parameters=(A1111, "tEXt")), ai=True, gen="Stable Diffusion (A1111-style)")
case("A1111 zTXt parameters", png(parameters=(A1111, "zTXt")), ai=True)
case("A1111 iTXt parameters", png(parameters=(A1111, "iTXt")), ai=True)
case("A1111 iTXt compressed", png(parameters=(A1111, "iTXtz")), ai=True)
case("ComfyUI prompt tEXt", png(prompt=(COMFY, "tEXt")), ai=True, gen="ComfyUI")
case("ComfyUI workflow zTXt", png(workflow=(COMFY_WF, "zTXt")), ai=True, gen="ComfyUI")
case("ComfyUI prompt iTXt compressed", png(prompt=(COMFY, "iTXtz")), ai=True, gen="ComfyUI")
case("InvokeAI iTXt", png(invokeai_metadata=(INVOKE, "iTXt")), ai=True, gen="InvokeAI")
case("sd-metadata tEXt", png(**{"sd-metadata": (INVOKE, "tEXt")}), ai=True, gen="InvokeAI")
case("Dream tEXt", png(Dream=('"a dog" -s 50 -S 42 -W 512', "tEXt")), ai=True, gen="InvokeAI")
case("NovelAI Comment", png(Software=("NovelAI", "tEXt"), Comment=(NOVELAI, "tEXt")), ai=True, gen="NovelAI")
case("Midjourney Description", png(Description=("a fox --v 6.1 --ar 16:9 Job ID: 0f2a1b3c-1111-2222-3333-444455556666", "tEXt")),
     ai=True, gen="Midjourney")
case("PNG Software CreatorTool", png(Software=("Stable Diffusion WebUI Forge", "tEXt")), ai=True)

# -- 2. IPTC / C2PA digitalSourceType, both URI prefixes
for base, label in (("http://cv.iptc.org/newscodes/digitalsourcetype/", "iptc"),
                    ("http://c2pa.org/digitalsourcetype/", "c2pa")):
    for term in ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia",
                 "algorithmicallyEnhanced", "algorithmicMedia", "dataDrivenMedia", "compositeSynthetic"):
        case(f"DST {label}/{term}", jpeg(xmp=xmp_dst(base, term)), ai=True, dst=term, dst_kind="ai")
    for term in ("digitalCapture", "computationalCapture", "negativeFilm", "positiveFilm", "print"):
        case(f"DST {label}/{term}", jpeg(xmp=xmp_dst(base, term)), ai=False, dst=term, dst_kind="camera")
    for term in ("humanEdits", "digitalCreation", "screenCapture", "virtualRecording",
                 "composite", "compositeCapture", "minorHumanEdits", "softwareImage", "digitalArt"):
        case(f"DST {label}/{term}", jpeg(xmp=xmp_dst(base, term)), ai=False, dst=term, dst_kind="neutral")
case("DST bare term, no URI", jpeg(xmp=xmp_dst("", "trainedAlgorithmicMedia")), ai=True, dst="trainedAlgorithmicMedia")
case("DST in PNG XMP", insert_png_chunk(png(), b"iTXt",
      b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp_dst("http://c2pa.org/digitalsourcetype/", "trainedAlgorithmicMedia").encode()),
     ai=True, dst="trainedAlgorithmicMedia")

# -- 3. EXIF
def exif_a1111(prefix: bytes, text: str, codec: str):
    ex = Image.Exif()
    ex[0x8769] = {0x9286: prefix + text.encode(codec)}
    return ex
case("EXIF UserComment UNICODE be", jpeg(exif=exif_a1111(b"UNICODE\x00", A1111, "utf-16-be")), ai=True)
case("EXIF UserComment UNICODE le", jpeg(exif=exif_a1111(b"UNICODE\x00", A1111, "utf-16-le")), ai=True)
case("EXIF UserComment ASCII", jpeg(exif=exif_a1111(b"ASCII\x00\x00\x00", A1111, "ascii")), ai=True)
case("EXIF UserComment no prefix", jpeg(exif=exif_a1111(b"\x00" * 8, A1111, "utf-8")), ai=True)
ex = Image.Exif(); ex[0x0131] = "Midjourney v6"
case("EXIF Software Midjourney", jpeg(exif=ex), ai=True, gen="Midjourney")
ex = Image.Exif(); ex[0x010E] = "a castle --v 6 Job ID: aabbccdd-1111-2222-3333-444455556666"
case("EXIF ImageDescription MJ job", jpeg(exif=ex), ai=True, gen="Midjourney")
ex = Image.Exif(); ex[0x9C9C] = "ComfyUI output".encode("utf-16-le")
case("EXIF XPComment", jpeg(exif=ex), ai=False, weak=True)
cam = Image.Exif()
cam[0x010F] = "Canon"; cam[0x0110] = "Canon EOS 5D Mark IV"
cam[0x8769] = {0x829A: IFDRational(1, 250), 0x829D: IFDRational(56, 10), 0x8827: 400,
               0x9003: "2019:07:04 11:22:33", 0xA434: "EF24-70mm f/2.8L II USM",
               0x920A: IFDRational(50, 1)}
cam[0x8825] = {1: "N", 2: (IFDRational(47), IFDRational(4), IFDRational(0))}
case("camera EXIF", jpeg(exif=cam), ai=False, camera=True)
cam2 = Image.Exif(); cam2[0x010F] = "Canon"; cam2[0x0110] = "EOS 5D"
cam2[0x8298] = "Prints available on canvas. Photo shows the Gemini spacecraft."
case("camera EXIF + risky caption", jpeg(exif=cam2), ai=False, camera=True, weak=True)

# -- 4. XMP variants
case("XMP CreatorTool Firefly", jpeg(xmp=XMP.format(ns='xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
     body='<xmp:CreatorTool>Adobe Firefly</xmp:CreatorTool>')), ai=True, gen="Adobe Firefly")
case("XMP CreatorTool attribute form", jpeg(xmp=XMP.format(
     ns='xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:CreatorTool="ComfyUI"', body='')), ai=True, gen="ComfyUI")
case("XMP softwareAgent in history", jpeg(xmp=XMP.format(
     ns='xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/" xmlns:stEvt="http://ns.adobe.com/xap/1.0/sType/ResourceEvent#"',
     body='<xmpMM:History><rdf:Seq><rdf:li stEvt:action="created" stEvt:softwareAgent="Midjourney Bot"/>'
          '</rdf:Seq></xmpMM:History>')), ai=True, gen="Midjourney")
case("XMP AIGC namespace", jpeg(xmp=XMP.format(ns='xmlns:AIGC="http://www.cac.gov.cn/aigc/1.0/"',
     body='<AIGC:Label>1</AIGC:Label><AIGC:ContentProducer>doubao</AIGC:ContentProducer>')), ai=True)
case("XMP GenAI namespace", jpeg(xmp=XMP.format(ns='xmlns:GenAI="http://ns.example.com/genai/1.0/"',
     body='<GenAI:model>imagen-3.0</GenAI:model>')), ai=True)
case("XMP camera props only", jpeg(xmp=XMP.format(
     ns='xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/" '
        'tiff:Make="NIKON CORPORATION" tiff:Model="NIKON D750" exif:ExposureTime="1/200" exif:FNumber="8/1"',
     body='')), ai=False, camera=True)
case("XMP dc:description mentions midjourney", jpeg(xmp=XMP.format(
     ns='xmlns:dc="http://purl.org/dc/elements/1.1/"',
     body='<dc:description><rdf:Alt><rdf:li xml:lang="x-default">An article about Midjourney</rdf:li>'
          '</rdf:Alt></dc:description>')), ai=False, weak=True)

# -- 5. C2PA
case("C2PA PNG caBX", insert_png_chunk(png(), b"caBX", jumbf()), ai=True, c2pa=True, gen="Adobe Firefly")
case("C2PA JPEG APP11", insert_jpeg_app11(jpeg(), jumbf()), ai=True, c2pa=True)
case("C2PA camera capture", insert_jpeg_app11(jpeg(), jumbf(b"Leica M11-P", b"digitalCapture")),
     ai=False, c2pa=True, dst="digitalCapture")
case("C2PA without label is not c2pa", insert_png_chunk(png(), b"caBX", b"\x00" * 64),
     c2pa=False, err=True)
case("C2PA structurally parsed", insert_png_chunk(png(), b"caBX", jumbf()),
     c2pa=True, jumbf=True, labels={"c2pa", "c2pa.assertions", "c2pa.actions", "c2pa.claim"})
case("unparseable JUMBF falls back to the string test",
     insert_jpeg_app11(jpeg(), b"\x00\x00\x00\x28jumb" + b"?" * 24 + b"c2pa.claim" + b"?" * 6),
     c2pa=True, jumbf=False)
case("C2PA in a BMFF jumb box", (lambda m: b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
     + struct.pack(">I", len(m) + 8) + b"jumb" + m)(
     jumbf()[8 + struct.unpack(">I", jumbf()[8:12])[0]:]), c2pa=True, jumbf=True)

# -- 6. false-positive guards
for caption in ("Fine art prints on canvas", "The Gemini spacecraft in orbit",
                "Una imagen de la playa", "Hard to grok this composite of two negatives",
                "A flux capacitor from the film", "Firefly season 1 episode 3",
                "Digital creation of the print shop"):
    ex = Image.Exif(); ex[0x010E] = caption
    case(f"caption: {caption[:28]}", jpeg(exif=ex), ai=False)

# -- 7. degradation
case("truncated png chunk", png(parameters=(A1111, "tEXt"))[:120], ai=False, err=True)
case("not an image", b"this is a plain text file, not an image at all" * 4, ai=False, err=True)
case("empty-ish", b"\x89PNG\r\n\x1a\n", ai=False)
zb = zlib.compress(b"A" * (64 << 20))
case("zTXt bomb", insert_png_chunk(png(), b"zTXt", b"parameters\x00\x00" + zb), ai=False, err=True)

# -- 8. WebP
def webp(xmp=None, exif=None):
    buf = io.BytesIO(); kw = {}
    if xmp: kw["xmp"] = xmp.encode()
    if exif is not None: kw["exif"] = exif
    BASE.save(buf, "WEBP", **kw); return buf.getvalue()
case("WebP XMP DST", webp(xmp=xmp_dst("http://cv.iptc.org/newscodes/digitalsourcetype/", "trainedAlgorithmicMedia")),
     ai=True, dst="trainedAlgorithmicMedia")
case("WebP EXIF UserComment", webp(exif=exif_a1111(b"UNICODE\x00", A1111, "utf-16-be")), ai=True)


# -- 9. PNG keyword case: forks disagree with the spec, we must not
case("PNG 'Parameters' capitalised", png(Parameters=(A1111, "tEXt")), ai=True)
case("PNG 'Prompt' capitalised", png(Prompt=(COMFY, "tEXt")), ai=True, gen="ComfyUI")
case("PNG 'WORKFLOW' upper", png(WORKFLOW=(COMFY_WF, "zTXt")), ai=True, gen="ComfyUI")
case("PNG 'DREAM' upper", png(DREAM=('"a dog" -s 50 -S 42', "tEXt")), ai=True, gen="InvokeAI")
case("PNG fooocus_scheme", png(fooocus_scheme=("Fooocus v2", "tEXt")), ai=True, gen="Stable Diffusion")
case("PNG sui_image_params", png(sui_image_params=(json.dumps(
     {"sui_image_params": {"prompt": "a dog", "model": "flux", "steps": 20,
                           "cfgscale": 1.0, "seed": 7, "sampler": "euler"}}), "tEXt")), ai=True)
case("ComfyUI graph in Comment", png(Comment=(COMFY, "tEXt")), ai=True, gen="ComfyUI")

# -- 10. JPEG COM segment
def jpeg_com(text, **kw):
    buf = io.BytesIO(); BASE.save(buf, "JPEG", comment=text.encode(), **kw); return buf.getvalue()
case("JPEG COM A1111", jpeg_com(A1111), ai=True, gen="Stable Diffusion (A1111-style)")
case("JPEG COM ComfyUI graph", jpeg_com(COMFY), ai=True, gen="ComfyUI")
case("JPEG COM Midjourney job", jpeg_com("a fox --v 6 Job ID: 1a2b3c4d-1111-2222-3333-444455556666"),
     ai=True, gen="Midjourney")
case("JPEG COM plain caption", jpeg_com("Shot on a Nikon at dusk"), ai=False)
case("JPEG COM naming a tool is weak only", jpeg_com("made with Midjourney"), ai=False, weak=True)

# -- 11. generator precedence: a placeholder must yield to a specific name
case("aiframework then Software", png(aiframework=("diffusers", "tEXt"), Software=("Midjourney", "tEXt")),
     ai=True, gen="Midjourney")

# -- 12. partial evidence (a generated region, not a generated image)
ex = Image.Exif(); ex[0x0131] = "Adobe Photoshop 25.0 (Generative Fill)"
case("EXIF Software generative fill", jpeg(exif=ex), ai=True, partial=True)
case("DST compositeWithTrainedAlgorithmicMedia is partial",
     jpeg(xmp=xmp_dst("http://c2pa.org/digitalsourcetype/", "compositeWithTrainedAlgorithmicMedia")),
     ai=True, partial=True)
case("DST trainedAlgorithmicMedia is not partial",
     jpeg(xmp=xmp_dst("http://c2pa.org/digitalsourcetype/", "trainedAlgorithmicMedia")),
     ai=True, partial=False)

# -- 13. vendors added for the 2024-25 crop, incl. CAC-labelled platforms
for soft, gen in (("Seedream 3.0", "ByteDance Seedream"), ("Jimeng AI", "ByteDance Seedream"),
                  ("Qwen-Image", "Qwen / Tongyi Wanxiang"), ("Hunyuan-DiT", "Tencent Hunyuan"),
                  ("Kling AI", "Kling"), ("CogView-4", "Zhipu CogView / ERNIE"),
                  ("RunwayML Gen-4", "Runway"), ("Luma Dream Machine", "Luma"),
                  ("Bing Image Creator", "Microsoft Designer / Bing"), ("Meta AI", "Meta AI"),
                  ("Civitai", "Civitai / TensorArt"), ("SwarmUI 0.9", "local diffusion UI"),
                  ("Freepik AI", "other hosted generator"), ("Stability AI", "Stability AI"),
                  ("ChatGPT", "DALL-E / OpenAI")):
    e = Image.Exif(); e[0x0131] = soft
    case(f"EXIF Software={soft}", jpeg(exif=e), ai=True, gen=gen)

# -- 14. ordinary words that must stay out of says_ai and out of .generator
for caption in ("An ideogram carved in stone", "A Kandinsky hanging in the hall",
                "Studio lights with diffusers", "I like to draw things on paper",
                "That car is a dream machine", "A neural filter bubble", "Runway 27L at dawn",
                "Luma of the display panel", "A meta discussion about ai ethics"):
    e = Image.Exif(); e[0x010E] = caption
    case(f"caption: {caption[:30]}", jpeg(exif=e), ai=False, no_gen=True)

# -- 14b. a tool field makes every pattern decisive, so mass-market apps that
# also handle ordinary photographs must not appear in the table at all
for soft in ("Kuaishou", "Doubao", "Tongyi", "Zhipu", "Runway", "Luma", "Sora", "Pika"):
    e = Image.Exif(); e[0x0131] = soft
    case(f"EXIF Software={soft} is not a generator mark", jpeg(exif=e), ai=False, no_gen=True)

# -- 14c. a job ticket in a caption is not a Midjourney job id
for cap in ("Archive Job ID: a1b2c3d4-5e6f transfer complete",
            "Job ID: 20240517-A4 shot on assignment"):
    e = Image.Exif(); e[0x010E] = cap
    case(f"caption: {cap[:30]}", jpeg(exif=e), ai=False)
e = Image.Exif(); e[0x010E] = "Job ID: a1b2c3d4-5e6f-7a8b-9c0d-e1f2a3b4c5d6"
case("full Midjourney job uuid still counts", jpeg(exif=e), ai=True, gen="Midjourney")
e = Image.Exif(); e[0x010E] = "sunset over the bay --ar 16:9 --v 6"
case("prompt flags in a caption are weak only", jpeg(exif=e), ai=False, weak=True)

# -- 14d. a generative edit must not be reported as a wholly generated image
for soft in ("Adobe Firefly Image 3 Model (generative fill)",
             "Adobe Photoshop 25.9 Generative Expand", "Adobe Photoshop Neural Filters"):
    e = Image.Exif(); e[0x0131] = soft
    case(f"partial: {soft[:34]}", jpeg(exif=e), ai=True, partial=True,
         gen="Adobe Firefly (generative edit)")
e = Image.Exif(); e[0x0131] = "Adobe Firefly"
case("plain Firefly is whole-image", jpeg(exif=e), ai=True, partial=False, gen="Adobe Firefly")

# -- 14e. pathological containers must be bounded, not walked byte by byte
# Walker bounds, asserted by yield count rather than wall clock: a timing
# assertion is flaky under pytest, and the property that matters is that a
# pathological container cannot make a walk longer than the cap.
from aidetect.metadata import _jpeg_segments, _png_chunks, _bmff_boxes, _riff_chunks, _MAX_BOXES
FILL = 8 << 20
for label, walk in (
        ("JPEG fill-byte run", lambda: _jpeg_segments(b"\xff\xd8" + b"\xff" * FILL, [])),
        ("PNG zero run", lambda: _png_chunks(b"\x89PNG\r\n\x1a\x0a" + b"\x00" * FILL, [])),
        ("PNG empty-chunk run", lambda: _png_chunks(
            b"\x89PNG\r\n\x1a\x0a" + b"\x00\x00\x00\x00aaaa\x00\x00\x00\x00" * 700000, [])),
        ("BMFF empty-box run", lambda: _bmff_boxes(b"\x00\x00\x00\x08aaaa" * 700000)),
        ("RIFF empty-chunk run", lambda: _riff_chunks(
            b"RIFF" + (FILL).to_bytes(4, "little") + b"WEBP" + b"aaaa\x00\x00\x00\x00" * 700000))):
    n = 0
    for _ in walk():
        n += 1
        if n > _MAX_BOXES:
            CASES.append((f"walker unbounded: {label} yielded past the cap", b"", {"impossible": True}))
            break

# a 64-bit box header is 16 bytes: sizes 8..15 must be refused, not walked
case("64-bit box with an 8-byte size", b"\x00\x00\x00\x18ftypavif" + b"\x00" * 8
     + b"\x00\x00\x00\x01jumb\x00\x00\x00\x00\x00\x00\x00\x0c" + b"\x00" * 64, c2pa=False)

# -- 15. EXIF recovered from the container when Pillow will not open the file
good = jpeg(exif=exif_a1111(b"UNICODE\x00", A1111, "utf-16-be"))
case("JPEG truncated mid-scan still yields EXIF", good[:len(good) // 2], ai=True, err=True)
case("WebP EXIF chunk", webp0 := (lambda: (lambda b: (BASE.save(b, "WEBP", exif=exif_a1111(
     b"ASCII\x00\x00\x00", A1111, "ascii")), b.getvalue())[1])(io.BytesIO()))(), ai=True)

# -- 16. path input (mmap) must agree with bytes input, incl. the empty file
import tempfile, os as _os
tmp = Path(tempfile.mkdtemp())
for nm, blob in (("a.png", png(parameters=(A1111, "tEXt"))), ("b.jpg", jpeg(exif=cam)),
                 ("c.png", insert_png_chunk(png(), b"caBX", jumbf())), ("empty.png", b"")):
    f = tmp / nm; f.write_bytes(blob)
    a, b = read_metadata(f), read_metadata(blob)
    if (a.ai_evidence, a.camera_evidence, a.generator, a.has_c2pa) !=        (b.ai_evidence, b.camera_evidence, b.generator, b.has_c2pa):
        CASES.append((f"path vs bytes: {nm}", b"", {"impossible": True}))

def _check(name: str, blob: bytes, exp: dict) -> list[str]:
    """Return every expectation this case violates, so one run names them all."""
    problems: list[str] = []
    try:
        v = read_metadata(blob)
    except Exception as exc:  # noqa: BLE001
        return [f"RAISED {type(exc).__name__}: {exc}"]
    if "ai" in exp and v.says_ai != exp["ai"]:
        problems.append(f"says_ai={v.says_ai} expected {exp['ai']} :: {v.ai_evidence} :: {v.errors}")
    if exp.get("gen") and v.generator != exp["gen"]:
        problems.append(f"generator={v.generator!r} expected {exp['gen']!r}")
    if "c2pa" in exp and v.has_c2pa != exp["c2pa"]:
        problems.append(f"has_c2pa={v.has_c2pa} expected {exp['c2pa']}")
    if exp.get("dst"):
        got = v.digital_source_type.term if v.digital_source_type else None
        if got != exp["dst"]:
            problems.append(f"dst={got!r} expected {exp['dst']!r}")
        if exp.get("dst_kind") and v.digital_source_type and \
                v.digital_source_type.kind != exp["dst_kind"]:
            problems.append(f"dst kind={v.digital_source_type.kind} expected {exp['dst_kind']}")
    if exp.get("camera") and not v.says_camera:
        problems.append(f"says_camera False, evidence={v.camera_evidence}")
    if exp.get("weak") and not v.weak_ai_evidence:
        problems.append("expected weak evidence, got none")
    if "partial" in exp and v.partial_ai != exp["partial"]:
        problems.append(f"partial_ai={v.partial_ai} expected {exp['partial']} :: {v.ai_evidence}")
    if exp.get("no_gen") and v.generator is not None:
        problems.append(f"generator={v.generator!r} set from ordinary prose")
    if "jumbf" in exp and v.c2pa and v.c2pa.structurally_confirmed != exp["jumbf"]:
        problems.append(f"structurally_confirmed={v.c2pa.structurally_confirmed} "
                        f"expected {exp['jumbf']}")
    if exp.get("labels") and (not v.c2pa or not exp["labels"] <= set(v.c2pa.labels)):
        problems.append(f"labels={v.c2pa.labels if v.c2pa else None} missing {exp['labels']}")
    if exp.get("impossible"):
        problems.append("path and bytes inputs disagree")
    if exp.get("err") and not v.errors:
        problems.append("expected an error note, got none")
    return problems


@pytest.mark.parametrize("name,blob,exp", CASES, ids=[c[0] for c in CASES])
def test_metadata_branch(name: str, blob: bytes, exp: dict) -> None:
    problems = _check(name, blob, exp)
    assert not problems, name + ": " + "; ".join(problems)


def test_absence_of_metadata_never_claims_ai() -> None:
    """The invariant the whole provenance channel rests on.

    Metadata can only ever argue *for* a conclusion. A file with nothing in it
    must not come back claiming either verdict, or a stripped upload would read
    as evidence of authenticity.
    """
    for blob in (png(), jpeg(), jpeg(quality=60)):
        v = read_metadata(blob)
        assert not v.says_ai
        assert not v.partial_ai
        assert not v.weak_ai_evidence
        assert not v.says_camera
