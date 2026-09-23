"""Provenance signals that live outside the pixels.

Generators and editors routinely leave their name in the file. When that
evidence is present it is far more reliable than any classifier, so it is worth
reading first -- but it is trivially stripped, so its absence proves nothing.
Every signal here is therefore reported as evidence *for* a conclusion, never
as evidence against one: nothing in this module can lower the probability that
an image is generated, and a file with no metadata at all yields an empty
verdict rather than a vote for "real".

Two rules keep precision near 1.0, which is the only reason this module is
worth running first:

* Containers are walked structurally (PNG chunks, JPEG markers, RIFF, ISO-BMFF)
  instead of grepped. Measured on this repo's own corpora, the naive test
  `b"c2pa" in blob or b"caBX" in blob or b"jumbf" in blob` fires on 4 of 300
  genuine DIV2K photographs where a structural walk finds nothing: those byte
  sequences simply occur inside compressed pixel data.
* Evidence is weighted by the role of the field it came from. A generator name
  in a field that names the producing tool is decisive; the same name in a
  caption is not, and goes to `weak_ai_evidence` where it cannot flip
  `says_ai`. "Imagen" is Spanish for "image" and "grok" is a verb.

Nothing here decodes pixels: 0.1-1.0 ms per file across this repo's corpora,
against 7-70 ms for the same work done through a full decode. Every container
walk is bounded, because the step size comes from the file itself: an
unbounded walk turned a malformed 2 GiB PNG into a 112 s hang.
"""

from __future__ import annotations

import html
import io
import json
import mmap
import os
import re
import struct
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from PIL import ExifTags, Image, JpegImagePlugin

# The whole file, as bytes or as a read-only mapping of one. Every container
# walker indexes and slices it; neither operation cares which it is.
_Buffer = bytes | mmap.mmap

# A generator name found in one of these fields is decisive: the field's whole
# purpose is to name the software that produced the file.
_ROLE_TOOL = "tool"
# Fields that exist only to record generation parameters. Anything found here
# is about the act of generating, not about the depicted subject.
_ROLE_PARAMS = "params"
# Captions, titles, authors, keywords. These describe the *subject*, so a
# generator name in them is a coincidence at least as often as a claim.
_ROLE_FREETEXT = "freetext"

# (pattern, canonical name, specific). "specific" means the pattern cannot
# plausibly appear in ordinary prose; only specific patterns are allowed to
# contribute anything at all from free text, and even then only weakly.
_GENERATOR_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = tuple(
    (re.compile(p, re.I), name, specific)
    for p, name, specific in (
        (r"midjourney|\bmj[\s_-]?v?[456]\b", "Midjourney", True),
        (r"stable[\s_-]?diffusion|automatic\s?1111|\bsd[\s_-]?webui\b|\bsdxl\b"
         r"|stable[\s_-]?cascade|\bfooocus\b|\bdreamstudio\b", "Stable Diffusion", True),
        (r"comfy[\s_-]?ui", "ComfyUI", True),
        (r"dall[\s_.·∙-]?e\b", "DALL-E / OpenAI", True),
        (r"\bopenai\b|\bgpt[\s-]?image\b", "DALL-E / OpenAI", False),
        (r"black[\s_-]?forest[\s_-]?labs|\bflux[\s._-]?(?:1[\s._-]?)?"
         r"(?:dev|schnell|pro|kontext|krea)\b", "FLUX / Black Forest Labs", True),
        (r"\bflux\b", "FLUX / Black Forest Labs", False),
        # Before the plain vendor names: Photoshop writes "Adobe Firefly ...
        # (generative fill)", and matching the vendor first would report a
        # retouched photograph as a wholly generated one.
        (r"generative\s+(?:fill|expand|remove|erase|recolou?r)|\bneural\s+filters\b",
         "Adobe Firefly (generative edit)", True),
        (r"adobe\s+firefly|\bfirefly[\s_-]?image\b", "Adobe Firefly", True),
        (r"\bfirefly\b", "Adobe Firefly", False),
        (r"\bimagen[\s._-]?\d|nano[\s_-]?banana", "Google Imagen / Gemini", True),
        (r"\bimagen\b|\bgemini\b|google\s+ai\b|\bvertex\s+ai\b", "Google Imagen / Gemini", False),
        (r"\bnovel\s?ai\b", "NovelAI", True),
        (r"leonardo[\s._-]?ai\b", "Leonardo.Ai", True),
        (r"\bideogram\b", "Ideogram", False),  # also an English word for a written symbol
        (r"\bkrea[\s._-]?ai\b", "Krea", True),
        (r"\bkrea\b", "Krea", False),
        (r"invoke[\s_-]?ai\b", "InvokeAI", True),
        (r"\brecraft[\s._-]?(?:ai|v\d)\b", "Recraft", True),
        (r"\bgrok[\s_-]?(?:imagine|\d)\b|\bx\.ai\b", "xAI Grok", True),
        (r"\bgrok\b|\bxai\b", "xAI Grok", False),
        (r"\bcanva\b[^.\n]{0,40}\b(?:magic|ai)\b", "Canva AI", True),
        # Named narrowly on purpose: a _ROLE_TOOL field makes every pattern
        # here decisive, so a mass-market app that merely exports photographs
        # (Kuaishou, Doubao, Tongyi, Zhipu) must not appear -- only the
        # generator brands themselves.
        (r"\bseedream\b|\bseededit\b|\bjimeng\b|byte\s?dance\s+(?:seed|ai)", "ByteDance Seedream", True),
        (r"\bqwen[\s._-]?image\b|tongyi\s*wanxiang|\bwanx\b", "Qwen / Tongyi Wanxiang", True),
        (r"\bhunyuan[\s._-]?(?:dit|image|\d)|\bhailuo\b", "Tencent Hunyuan", True),
        (r"\bkling[\s._-]?(?:ai|\d)", "Kling", True),
        (r"\bcogview\b|\bernie[\s._-]?vilg\b", "Zhipu CogView / ERNIE", True),
        (r"runway\s?ml\b|\brunway\s+gen[\s._-]?\d", "Runway", True),
        (r"luma\s?labs\b|luma[\s._-]?(?:ai|dream)", "Luma", True),
        (r"bing\s+image\s+creator|microsoft\s+designer|image\s+creator\s+from\s+designer",
         "Microsoft Designer / Bing", True),
        (r"\bmeta[\s._-]?ai\b|\bemu[\s._-]?(?:image|edit)\b|imagine\.meta", "Meta AI", True),
        (r"\bcivitai\b|\btensor\.?art\b|\bseaart\b|\bshakker\b", "Civitai / TensorArt", True),
        (r"\bswarm\s?ui\b|stable\s?swarm|\bdrawthings\b|diffusion\s?bee|\bsd\.next\b"
         r"|\bautomatic1111\b", "local diffusion UI", True),
        (r"\bfreepik\b|getimg\.ai|\bnightcafe\b|\bartbreeder\b|\bcraiyon\b|\bstarryai\b"
         r"|\bwombo\b|\bpika\s?labs\b|\bmage\.space\b", "other hosted generator", True),
        (r"\bstability[\s._-]?ai\b|\bstable[\s._-]?image[\s._-]?(?:core|ultra)\b",
         "Stability AI", True),
        (r"\bchatgpt\b|\bsora[\s._-]?\d\b", "DALL-E / OpenAI", True),
        (r"\bdeci[\s_-]?diffusion\b|\bplayground\s?v?\d", "other diffusion model", True),
        # Both are ordinary words: a photo of a Kandinsky, a studio light diffuser.
        (r"\bdiffusers\b|\bkandinsky\b", "other diffusion model", False),
    )
)

# Names that only say "something generative", so a later, specific match is
# allowed to replace them. Without this the first weak hit would lock the
# field: an 'aiframework' PNG chunk would outrank a later "Software: Midjourney".
_PLACEHOLDER_GENERATORS = frozenset((
    "other diffusion model", "other hosted generator", "local diffusion UI",
    "unknown diffusion pipeline", "unknown (AIGC-labelled)",
))

# Generator names that assert a synthesised region inside an otherwise real
# photograph. Callers that need "wholly generated" must check `partial_ai`.
_PARTIAL_GENERATORS = frozenset(("Adobe Firefly (generative edit)",))
_PARTIAL_MARK = " (generated element, not whole image)"

# IPTC/C2PA digital source vocabulary. Both registries publish the same terms
# under different URIs, and writers cite either.
_DST_URI = re.compile(
    r"https?://(?:cv\.iptc\.org/newscodes/digitalsourcetype/|c2pa\.org/digitalsourcetype/)"
    r"([A-Za-z]+)")

_DST_AI = {
    "trainedalgorithmicmedia": "trainedAlgorithmicMedia",
    "compositewithtrainedalgorithmicmedia": "compositeWithTrainedAlgorithmicMedia",
    "algorithmicallyenhanced": "algorithmicallyEnhanced",
    "algorithmicmedia": "algorithmicMedia",
    "datadrivenmedia": "dataDrivenMedia",
    "compositesynthetic": "compositeSynthetic",
}
_DST_CAMERA = {
    "digitalcapture": "digitalCapture",
    "computationalcapture": "computationalCapture",
    "negativefilm": "negativeFilm",
    "positivefilm": "positiveFilm",
    "print": "print",
}
_DST_NEUTRAL = {
    "humanedits": "humanEdits",
    "digitalcreation": "digitalCreation",
    "screencapture": "screenCapture",
    "virtualrecording": "virtualRecording",
    "composite": "composite",
    "compositecapture": "compositeCapture",
    "minorhumanedits": "minorHumanEdits",
    "softwareimage": "softwareImage",
    "digitalart": "digitalArt",
}
# Withdrawn from the IPTC scheme but still written by deployed tools.
_DST_RETIRED = {"algorithmicmedia", "minorhumanedits", "softwareimage", "digitalart"}
# Terms that assert a generated *element*, not a wholly generated image.
_DST_PARTIAL = {"compositewithtrainedalgorithmicmedia", "algorithmicallyenhanced", "compositesynthetic"}

# China's CAC labelling rule makes platforms tag generated media in XMP; the
# namespace prefix rather than the value is what carries the claim.
_AIGC_NAME = re.compile(r"(?:^|[:_.-])(?:aigc|genai|gen_ai|generativeai|aigeneratedcontent)", re.I)
# Values that turn such a property into a denial rather than a claim.
_DENIALS = frozenset(("false", "0", "no", "none", "off", "n", "unknown", ""))

_A1111_KEYS = ("Steps:", "Sampler:", "CFG scale:", "Seed:", "Model hash:",
               "Denoising strength:", "Schedule type:", "Negative prompt:", "Clip skip:")

_GEN_JSON_SAMPLING_KEYS = {"steps", "sampler", "seed", "cfg_scale", "uc", "negative_prompt",
                           "noise_schedule", "generation_mode", "sm_dyn", "scheduler"}
_GEN_JSON_KEYS = {"scale", "strength", "model", "prompt", "app_version"}
# Midjourney keeps its job id and prompt flags in an otherwise free-text field.
# The id must be a whole UUID: press and asset-management pipelines stamp
# "Job ID: <ticket>" into the caption of genuine photographs, and a prefix-only
# match turns every one of those into a false positive.
_MIDJOURNEY_JOB = re.compile(
    r"Job ID:\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_MIDJOURNEY_FLAGS = re.compile(r"--(?:v|ar|q|style|chaos|niji|stylize|sref|cref)\s+\S", re.I)

# Keys whose *name* is a vendor marker: nothing but that tool writes them, so
# the key alone is evidence even when the payload is unreadable.
# Lower-cased throughout: PNG keywords are case-sensitive by spec, but forks
# disagree on it ("Parameters", "Prompt"), and no ordinary keyword collides
# with these under case folding.
_VENDOR_PNG_KEYS = {
    "invokeai_metadata": "InvokeAI",
    "invokeai_graph": "InvokeAI",
    "sd-metadata": "InvokeAI",
    "fooocus_scheme": "Stable Diffusion",
    "aiframework": "unknown diffusion pipeline",
}
# InvokeAI's legacy chunk is keyed on an ordinary English word, so unlike every
# other vendor key it cannot count on its own. Its payload is a dream.py
# command line -- a quoted prompt followed by single-letter flags -- and two of
# those flags is a shape no caption arrives at by accident.
_INVOKEAI_DREAM = re.compile(r"(?:^|\s)-[sSWHCAfInGUv]\s*[\d.]")
# Keys that are ordinary English words. A1111 and ComfyUI write these, but so
# could anything else, so they only count when the payload has the expected
# shape.
_PARAM_PNG_KEYS = ("parameters", "prompt", "workflow", "generation_data", "sui_image_params")
_TOOL_PNG_KEYS = ("software", "source")
_FREETEXT_PNG_KEYS = ("title", "author", "description", "copyright", "disclaimer",
                      "warning", "comment", "dream")

_EXIF_TAG = {v: k for k, v in ExifTags.TAGS.items()}
_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825

# Tag id -> canonical name, kept explicit because these ids live in three
# different IFDs and Pillow's reverse map is ambiguous across them.
_CAMERA_TAGS_IFD0 = {0x010F: "Make", 0x0110: "Model"}
_CAMERA_TAGS_EXIF = {
    0x829A: "ExposureTime", 0x829D: "FNumber", 0x8827: "ISOSpeedRatings",
    0x920A: "FocalLength", 0x9003: "DateTimeOriginal", 0xA434: "LensModel",
    0x9201: "ShutterSpeedValue", 0x9202: "ApertureValue", 0xA433: "LensMake",
    0x927C: "MakerNote", 0xA431: "BodySerialNumber", 0x9291: "SubsecTimeOriginal",
}
_TEXT_TAGS_IFD0 = {0x010E: "ImageDescription", 0x0131: "Software",
                   0x013B: "Artist", 0x8298: "Copyright"}
# Windows writes these as UTF-16LE in IFD0.
_XP_TAGS = {0x9C9B: "XPTitle", 0x9C9C: "XPComment", 0x9C9D: "XPAuthor",
            0x9C9E: "XPKeywords", 0x9C9F: "XPSubject"}

_IIM_FIELDS = {5: "ObjectName", 25: "Keywords", 65: "OriginatingProgram",
               70: "ProgramVersion", 80: "By-line", 110: "Credit", 115: "Source",
               116: "CopyrightNotice", 120: "Caption"}
# 2:65 names the producing application; the rest describe the subject.
_IIM_TOOL_FIELDS = {"OriginatingProgram", "ProgramVersion"}

_PNG_META_CHUNKS = frozenset((b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"caBX"))

_XMP_PACKET = re.compile(rb"<x:xmpmeta[^>]*>.{0,1000000}?</x:xmpmeta>", re.DOTALL)
_XMP_APP1 = b"http://ns.adobe.com/xap/1.0/\x00"
_XMP_APP1_EXT = b"http://ns.adobe.com/xmp/extension/\x00"

_C2PA_ACTIONS = {
    "c2pa.created", "c2pa.opened", "c2pa.placed", "c2pa.edited", "c2pa.removed",
    "c2pa.cropped", "c2pa.resized", "c2pa.filtered", "c2pa.drawing", "c2pa.converted",
    "c2pa.transcoded", "c2pa.repackaged", "c2pa.published", "c2pa.printed",
    "c2pa.color_adjustments", "c2pa.orientation", "c2pa.managed", "c2pa.unknown",
}
_C2PA_LABELS = {
    "c2pa.actions", "c2pa.actions.v2", "c2pa.assertions", "c2pa.claim", "c2pa.claim.v2",
    "c2pa.cloud-data", "c2pa.credentials", "c2pa.databoxes", "c2pa.depthmap",
    "c2pa.endorsement", "c2pa.hash.bmff", "c2pa.hash.bmff.v2", "c2pa.hash.boxes",
    "c2pa.hash.data", "c2pa.ingredient", "c2pa.ingredient.v2", "c2pa.ingredient.v3",
    "c2pa.manifest", "c2pa.metadata", "c2pa.signature", "c2pa.soft-binding",
    "c2pa.thumbnail.claim", "c2pa.thumbnail.ingredient", "c2pa.time-stamp",
    "c2pa.training-mining",
}
_C2PA_VOCAB = tuple(sorted(_C2PA_ACTIONS | _C2PA_LABELS, key=len, reverse=True))
_C2PA_TOKEN = re.compile(rb"c2pa\.[a-zA-Z][a-zA-Z0-9_.-]{2,80}")

# Every JUMBF content-type UUID the C2PA spec defines is four ASCII bytes
# naming the box kind followed by this fixed suffix. Matching it is what turns
# "the bytes c2pa occur in this file" into "this file carries a manifest
# store", and it is the reason nothing here needs a C2PA library.
_JUMBF_UUID_SUFFIX = bytes.fromhex("00110010800000aa00389b71")
_JUMBF_MANIFEST_TAGS = frozenset((b"c2pa", b"c2ma"))

_MAX_RAW = 400
_MAX_XMP = 4000
# Every container walker steps forward by a length the file itself supplies, so
# a file of zeros is a chunk stream of 2^31/12 empty chunks: a malformed 2 GiB
# PNG cost 112 s before this cap. No real file comes near it -- a 2 GiB PNG
# written with 64 KiB IDATs has 32k chunks.
_MAX_BOXES = 1 << 16
# PNG caps a text keyword at 79 bytes; anything longer is malformed and must
# not reach a dict key or an error message at the file's chosen length.
_MAX_PNG_KEYWORD = 79
# XMP packets sit at the head of a container or appended past its end, so a
# window at each end bounds the one scan here that is linear in file size.
_XMP_SCAN_WINDOW = 8 << 20
# zlib bomb guard for PNG zTXt/iTXt: no legitimate generation blob is this big.
_MAX_INFLATE = 16 << 20

_IJG_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99)
_IJG_CHROMA = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99)


@dataclass(frozen=True)
class DigitalSourceType:
    """An IPTC/C2PA digitalSourceType term, as written and as interpreted."""

    term: str
    kind: str  # "ai" | "camera" | "neutral"
    where: str
    retired: bool = False
    partial: bool = False  # asserts a generated element, not a generated image


@dataclass
class C2paEvidence:
    """A C2PA manifest was found. Nothing about it has been verified."""

    container: str
    manifest_bytes: int = 0
    claim_generator: str | None = None
    actions: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    digital_source_type: str | None = None
    # The JUMBF box structure parsed, and a C2PA content-type UUID was found in
    # it. False means only the string "c2pa" was seen where a manifest belongs.
    structurally_confirmed: bool = False
    # Signature checking needs the trust list and a crypto stack; we do not do
    # it, and callers must not read presence as authenticity.
    cryptographically_validated: bool = False


@dataclass
class JpegStructure:
    """Container-level JPEG statistics. A weak feature, never a verdict.

    Camera and generator JPEGs differ here in distribution only: phones and
    cameras ship tuned quantisation tables, while generator pipelines save
    through libjpeg/Pillow defaults. Any single file can look like either.
    """

    estimated_quality: int | None = None
    ijg_quality: int | None = None  # set only on an exact libjpeg-default table match
    n_quant_tables: int = 0
    subsampling: str | None = None
    progressive: bool = False
    has_jfif: bool = False
    has_adobe: bool = False
    app_segments: list[str] = field(default_factory=list)

    @property
    def library_default_tables(self) -> bool:
        return self.ijg_quality is not None


@dataclass
class MetadataVerdict:
    """What the file's own metadata says about how it was made."""

    ai_evidence: list[str] = field(default_factory=list)
    camera_evidence: list[str] = field(default_factory=list)
    weak_ai_evidence: list[str] = field(default_factory=list)
    generator: str | None = None
    has_c2pa: bool = False
    c2pa: C2paEvidence | None = None
    digital_source_type: DigitalSourceType | None = None
    jpeg: JpegStructure | None = None
    errors: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def says_ai(self) -> bool:
        return bool(self.ai_evidence)

    @property
    def partial_ai(self) -> bool:
        """Every AI claim found concerns a generated region, not a whole image.

        A generative fill on a photograph and a wholly synthesised picture both
        set `says_ai`, and a caller that grades "is this a real photograph?"
        needs to tell them apart.
        """
        return bool(self.ai_evidence) and all(_PARTIAL_MARK in e for e in self.ai_evidence)

    @property
    def says_camera(self) -> bool:
        """True when the file carries capture-shaped metadata.

        Weak evidence: EXIF is easy to forge and easy to copy onto a generated
        file. Treated as a nudge, never as proof.
        """
        return len(self.camera_evidence) >= 2 and not self.ai_evidence

    def describe(self) -> str:
        if self.ai_evidence:
            return "; ".join(self.ai_evidence)
        if self.weak_ai_evidence:
            return "weak only: " + "; ".join(self.weak_ai_evidence)
        return "no provenance evidence"


def _clip(text: str, limit: int = _MAX_RAW) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


# -- container walkers ----------------------------------------------------
#
# Each of these yields only what the container structure says is there. That
# is the whole defence against matching a marker inside compressed pixel data.

def _sniff_format(blob: _Buffer) -> str:
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if blob[:2] == b"\xff\xd8":
        return "JPEG"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "WEBP"
    if blob[4:8] == b"ftyp":
        return "BMFF"
    if blob[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return ""


def _png_chunks(blob: _Buffer, errors: list[str], want: frozenset[bytes] | None = None):
    """Yield (type, payload) for the chunks in `want`.

    Filtering inside the walk matters: a 4 MPix PNG is several hundred IDAT
    chunks, and slicing each one copies the whole file for nothing.
    """
    pos, n = 8, len(blob)
    for _ in range(_MAX_BOXES):
        if pos + 12 > n:
            return
        length = struct.unpack(">I", blob[pos:pos + 4])[0]
        ctype = bytes(blob[pos + 4:pos + 8])
        if not ctype.isalpha():
            # PNG chunk types are four ASCII letters. Anything else means we
            # are no longer looking at a chunk stream, and stepping on would
            # walk the whole file a dozen bytes at a time.
            errors.append(f"PNG chunk stream breaks down at offset {pos}")
            return
        if length > n - pos - 12:
            errors.append(f"PNG chunk {ctype.decode('latin-1')} runs past end of file")
            return
        if want is None or ctype in want:
            yield ctype, blob[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IEND":
            return
    errors.append(f"PNG has more than {_MAX_BOXES} chunks; stopped walking")


def _jpeg_segments(blob: _Buffer, errors: list[str]):
    pos, n = 2, len(blob)
    for _ in range(_MAX_BOXES):
        if pos + 4 > n:
            return
        if blob[pos] != 0xFF:
            errors.append(f"JPEG marker expected at offset {pos}")
            return
        marker = blob[pos + 1]
        if marker == 0xFF:  # fill bytes, legal in any number before a marker
            pos += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", blob[pos + 2:pos + 4])[0]
        if length < 2 or pos + 2 + length > n:
            errors.append("JPEG segment length runs past end of file")
            return
        yield marker, blob[pos + 4:pos + 2 + length]
        if marker == 0xDA:  # entropy-coded data follows; metadata is all before it
            return
        pos += 2 + length
    errors.append(f"JPEG has more than {_MAX_BOXES} segments; stopped walking")


def _riff_chunks(blob: _Buffer):
    pos, n = 12, min(len(blob), struct.unpack("<I", blob[4:8])[0] + 8)
    for _ in range(_MAX_BOXES):
        if pos + 8 > n:
            return
        fourcc = blob[pos:pos + 4]
        size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        if pos + 8 + size > n:
            return
        yield fourcc, blob[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)


def _bmff_boxes(blob: _Buffer, start: int = 0, end: int | None = None, depth: int = 0):
    """Walk ISO-BMFF boxes, descending only into the containers that can hold
    a manifest. HEIC/AVIF put C2PA in a 'jumb' box under 'meta'."""
    pos = start
    end = len(blob) if end is None else end
    for _ in range(_MAX_BOXES):
        if pos + 8 > end:
            return
        size = struct.unpack(">I", blob[pos:pos + 4])[0]
        btype = blob[pos + 4:pos + 8]
        body = pos + 8
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", blob[pos + 8:pos + 16])[0]
            body, header = pos + 16, 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield btype, blob[body:pos + size]
        if depth < 3 and btype in (b"meta", b"moov", b"udta", b"iprp", b"trak"):
            # 'meta' is a FullBox: four bytes of version/flags before children.
            skip = 4 if btype == b"meta" else 0
            yield from _bmff_boxes(blob, body + skip, pos + size, depth + 1)
        pos += size


# -- text decoding --------------------------------------------------------

def _inflate(data: bytes) -> bytes:
    obj = zlib.decompressobj()
    out = obj.decompress(data, _MAX_INFLATE)
    if obj.unconsumed_tail:
        raise ValueError("compressed text chunk exceeds the inflate cap")
    return out


@dataclass
class _PngParts:
    text: dict[str, tuple[str, str]] = field(default_factory=dict)
    exif: bytes | None = None
    c2pa: bytes | None = None


def _png_parts(blob: _Buffer, errors: list[str]) -> _PngParts:
    """One pass over the chunk stream for text, EXIF and the C2PA box.

    Text is decoded here rather than read from Pillow's `info` because Pillow
    only surfaces chunks that precede IDAT and hides which chunk type carried
    them; EXIF is read here because `PngImageFile.getexif()` calls `load()`,
    decoding every pixel in the file to reach a possible trailing eXIf chunk.
    """
    parts = _PngParts()
    out = parts.text
    for ctype, data in _png_chunks(blob, errors, _PNG_META_CHUNKS):
        if ctype == b"eXIf":
            parts.exif = data
            continue
        if ctype == b"caBX":
            # The chunk type is registered by the C2PA spec and used for
            # nothing else, so whatever is in it is the manifest candidate;
            # whether it really is one is decided by the JUMBF walk.
            parts.c2pa = data
            continue
        # All three text chunks open with keyword + NUL, so the keyword is read
        # once, before anything is decoded. The spec caps it at 79 bytes; a
        # longer one is malformed and would otherwise reach a dict key and an
        # error message at whatever length the file chose.
        kind = ctype.decode("latin-1")
        key, sep, rest = data.partition(b"\x00")
        if not sep:
            errors.append(f"{kind} chunk has no keyword separator")
            continue
        if len(key) > _MAX_PNG_KEYWORD:
            errors.append(f"{kind} keyword is {len(key)} bytes, over the 79-byte limit; ignored")
            continue
        name = key.decode("latin-1", "replace")
        try:
            if ctype == b"tEXt":
                text = rest.decode("latin-1")
            elif ctype == b"zTXt":
                if not rest or rest[0] != 0:
                    errors.append(f"zTXt {name}: unknown compression method")
                    continue
                text = _inflate(rest[1:]).decode("latin-1")
            elif ctype == b"iTXt":
                if len(rest) < 2:
                    continue
                compressed, method = rest[0], rest[1]
                _lang, _, rest = rest[2:].partition(b"\x00")
                _translated, _, payload = rest.partition(b"\x00")
                if compressed:
                    if method != 0:
                        errors.append(f"iTXt {name}: unknown compression method")
                        continue
                    payload = _inflate(payload)
                text = payload.decode("utf-8", "replace")
            else:
                continue
        except (zlib.error, ValueError) as exc:
            errors.append(f"{kind} chunk '{name}' is unreadable: {exc}")
            continue
        if name not in out:  # first chunk wins, as PNG readers do
            out[name] = (text, kind)
    return parts


def _decode_user_comment(raw: bytes) -> str:
    """Strip the 8-byte EXIF character-code prefix before decoding.

    A1111 writes its whole parameter string here for JPEG and WebP, and the
    prefix is part of the EXIF spec, not part of the text.
    """
    if len(raw) < 8:
        return raw.decode("utf-8", "replace")
    head, body = raw[:8], raw[8:]
    if head.startswith(b"ASCII"):
        return body.decode("ascii", "replace")
    if head.startswith(b"UNICODE"):
        # The spec says UCS-2 in the file's byte order and writers disagree, so
        # the byte order has to be inferred. Counting printable characters does
        # not do it: wrong-endian Latin text decodes to assigned CJK code
        # points, which are printable, so both orders score 1.0 and the tie
        # always fell to the first one tried. Where the padding zeros land says
        # it outright, and these fields carry Latin text.
        if body[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return body.decode("utf-16", "replace")
        even = sum(body[i] == 0 for i in range(0, len(body) - 1, 2))
        odd = sum(body[i] == 0 for i in range(1, len(body), 2))
        return body.decode("utf-16-be" if even >= odd else "utf-16-le", "replace")
    if head.startswith(b"JIS"):
        return body.decode("shift_jis", "replace")
    if head == b"\x00" * 8:
        return body.decode("utf-8", "replace")
    return raw.decode("utf-8", "replace")


def _exif_str(value: object, tag: int = 0) -> str:
    if isinstance(value, bytes):
        if tag == 0x9286:
            return _decode_user_comment(value).strip("\x00").strip()
        if tag in _XP_TAGS:
            return value.decode("utf-16-le", "replace").strip("\x00").strip()
        return value.decode("utf-8", "replace").strip("\x00").strip()
    if isinstance(value, str):
        return value.strip("\x00").strip()
    return ""


def _xmp_values(text: str, local: str) -> list[str]:
    """Values of `local` regardless of namespace prefix, attribute or element.

    Matching is case-sensitive because XML names are: `dc:description` and
    `rdf:Description` differ only in case, and folding them together swallows
    an entire rdf block as if it were a caption.
    """
    if local not in text:  # the regexes below backtrack; this check does not
        return []
    out: list[str] = []
    for m in re.finditer(rf'(?:^|[\s"])(?:[\w.-]+:)?{re.escape(local)}\s*=\s*"([^"]*)"', text):
        out.append(m.group(1))
    for m in re.finditer(rf'<[\w.-]*:?{re.escape(local)}(?:\s[^>]*)?>(.*?)</[\w.-]*:?{re.escape(local)}>',
                         text, re.DOTALL):
        out.append(re.sub(r"<[^>]+>", " ", m.group(1)))
    return [v for v in (html.unescape(x).strip() for x in out) if v]


# -- structured generator signatures --------------------------------------

def _is_a1111(text: str) -> bool:
    # Two keys, because a caption can mention "Seed:" on its own.
    return sum(k in text for k in _A1111_KEYS) >= 2


def _is_comfy_graph(text: str) -> bool:
    if "class_type" not in text and '"nodes"' not in text:
        return False
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        # Truncated graphs are common once a file has been through a tool that
        # caps text-chunk length, and "class_type" belongs to nothing else.
        return '"class_type"' in text
    if isinstance(parsed, dict):
        if any(isinstance(v, dict) and "class_type" in v for v in parsed.values()):
            return True
        nodes = parsed.get("nodes")
        return isinstance(nodes, list) and any(
            isinstance(n, dict) and "type" in n and "widgets_values" in n for n in nodes)
    return False


def _json_generator_keys(text: str) -> list[str]:
    """Generation-parameter keys in a JSON blob (NovelAI, InvokeAI, Draw Things).

    Returns nothing unless one of the keys is specific to sampling, so that a
    JSON caption carrying `model`/`prompt`/`scale` -- words an asset-management
    system uses too -- cannot on its own be read as a generation record.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        return []
    if not isinstance(parsed, dict):
        return []
    flat = dict(parsed)
    for value in list(parsed.values()):
        if isinstance(value, dict):
            flat.update(value)
    found = sorted(k for k in flat if isinstance(k, str)
                   and k.lower() in _GEN_JSON_KEYS | _GEN_JSON_SAMPLING_KEYS)
    return found if any(k.lower() in _GEN_JSON_SAMPLING_KEYS for k in found) else []


# -- evidence bookkeeping -------------------------------------------------

def _note_generator(v: MetadataVerdict, name: str) -> None:
    if v.generator is None or (v.generator in _PLACEHOLDER_GENERATORS
                               and name not in _PLACEHOLDER_GENERATORS):
        v.generator = name


def _note_ai(v: MetadataVerdict, text: str, partial: bool = False) -> None:
    v.ai_evidence.append(text + (_PARTIAL_MARK if partial else ""))


def _scan_names(v: MetadataVerdict, where: str, text: str, role: str) -> None:
    """Match generator names, with the field's role deciding the weight."""
    if not text:
        return
    for pattern, name, specific in _GENERATOR_PATTERNS:
        if not pattern.search(text):
            continue
        if role == _ROLE_TOOL or (role == _ROLE_PARAMS and specific):
            # A parameter blob contains the prompt, and a prompt is prose: only
            # a name that cannot occur in prose may be decisive there. A field
            # whose sole purpose is to name the tool is decisive either way.
            _note_generator(v, name)
            _note_ai(v, f"{where} names {name}", name in _PARTIAL_GENERATORS)
        elif specific:
            _note_generator(v, name)
            v.weak_ai_evidence.append(f"{where} mentions {name}")
        else:
            v.weak_ai_evidence.append(f"{where} mentions {name} (ambiguous wording)")
        return


def _scan_freetext(v: MetadataVerdict, where: str, text: str, what: str,
                   fallback: str = _ROLE_FREETEXT) -> None:
    """A field that tools dump whole generation records into.

    The structured shapes are tested first and count in full wherever they are
    found: nothing writes an A1111 parameter block or a node graph into a
    caption by accident. Only when none matches does the text go to the name
    scan, and there `fallback` decides what a bare generator name is worth --
    decisive in a parameter field, a weak hint in a caption.
    """
    if _is_a1111(text):
        _note_generator(v, "Stable Diffusion (A1111-style)")
        _note_ai(v, f"{what} holds A1111 generation settings")
    elif _is_comfy_graph(text):
        _note_generator(v, "ComfyUI")
        _note_ai(v, f"{what} holds a ComfyUI node graph")
    elif len(_json_generator_keys(text)) >= 3:
        _note_generator(v, "unknown diffusion pipeline")
        _note_ai(v, f"{what} holds generation parameters")
    elif _MIDJOURNEY_JOB.search(text):
        _note_generator(v, "Midjourney")
        _note_ai(v, f"{what} holds a Midjourney job id")
    elif _MIDJOURNEY_FLAGS.search(text):
        _note_generator(v, "Midjourney")
        if fallback == _ROLE_FREETEXT:
            v.weak_ai_evidence.append(f"{what} carries Midjourney prompt flags")
        else:
            _note_ai(v, f"{what} holds Midjourney prompt flags")
    else:
        _scan_names(v, where, text, fallback)


def _scan_dst(v: MetadataVerdict, where: str, text: str) -> None:
    """IPTC/C2PA digitalSourceType, which only counts when it is a real term.

    The URI prefix is required unless the value came from a DigitalSourceType
    field, so an image whose caption says "print" or "composite" is untouched.
    """
    found: list[str] = [m.group(1) for m in _DST_URI.finditer(text)]
    for value in _xmp_values(text, "DigitalSourceType"):
        bare = value.rsplit("/", 1)[-1].strip()
        if bare:
            found.append(bare)
    for term in found:
        key = term.lower()
        if key in _DST_AI:
            kind, canon = "ai", _DST_AI[key]
        elif key in _DST_CAMERA:
            kind, canon = "camera", _DST_CAMERA[key]
        elif key in _DST_NEUTRAL:
            kind, canon = "neutral", _DST_NEUTRAL[key]
        else:
            v.errors.append(f"{where}: unknown digitalSourceType '{_clip(term, 60)}'")
            continue
        dst = DigitalSourceType(term=canon, kind=kind, where=where,
                                retired=key in _DST_RETIRED, partial=key in _DST_PARTIAL)
        if v.digital_source_type is None or (kind == "ai" and v.digital_source_type.kind != "ai"):
            v.digital_source_type = dst
        if kind == "ai":
            _note_ai(v, f"{where} declares digitalSourceType={canon}", dst.partial)
        elif kind == "camera":
            v.camera_evidence.append(f"digitalSourceType:{canon}")


# -- per-container scanners -----------------------------------------------

def _scan_png(v: MetadataVerdict, blob: _Buffer) -> tuple[list[str], bytes | None, bytes | None]:
    """Scan the PNG chunk stream, returning (xmp packets, c2pa box, exif block)."""
    parts = _png_parts(blob, v.errors)
    xmp: list[str] = []

    for key, (value, chunk) in parts.text.items():
        if not value.strip():
            continue
        name = key.lower()
        if name == "xml:com.adobe.xmp":
            xmp.append(value)
            continue
        v.raw[f"png:{key}"] = _clip(value)
        v.raw.setdefault("png:chunk_types", {})[key] = chunk

        vendor = _VENDOR_PNG_KEYS.get(name)
        if name == "dream" and len(_INVOKEAI_DREAM.findall(value)) >= 2:
            vendor = "InvokeAI"
        if vendor:
            _note_generator(v, vendor)
            _note_ai(v, f"PNG '{key}' chunk is a {vendor} record")
            _scan_names(v, f"PNG:{key}", value, _ROLE_PARAMS)
        elif name in _PARAM_PNG_KEYS:
            keys = _json_generator_keys(value)
            if _is_comfy_graph(value):
                # No name scan: the checkpoint filename in the graph would
                # otherwise rename the generator after the UI that ran it.
                _note_generator(v, "ComfyUI")
                _note_ai(v, f"PNG '{key}' chunk holds a ComfyUI node graph")
            elif _is_a1111(value):
                _note_generator(v, "Stable Diffusion (A1111-style)")
                _note_ai(v, f"PNG '{key}' chunk holds A1111 generation settings")
                _scan_names(v, f"PNG:{key}", value, _ROLE_PARAMS)
            else:
                if len(keys) >= 3:
                    _note_generator(v, "unknown diffusion pipeline")
                    _note_ai(v, f"PNG '{key}' chunk holds generation parameters "
                                f"({', '.join(keys[:5])})")
                _scan_names(v, f"PNG:{key}", value, _ROLE_PARAMS)
        elif name in _TOOL_PNG_KEYS:
            _scan_names(v, f"PNG:{key}", value, _ROLE_TOOL)
        elif name in _FREETEXT_PNG_KEYS:
            _scan_freetext(v, f"PNG:{key}", value, f"PNG '{key}' chunk")
        _scan_dst(v, f"PNG:{key}", value)
    return xmp, parts.c2pa, parts.exif


def _scan_jpeg(v: MetadataVerdict,
               blob: _Buffer) -> tuple[list[str], bytes | None, list[str], bytes | None]:
    xmp: list[str] = []
    c2pa = bytearray()
    apps: list[str] = []
    exif: bytes | None = None
    for marker, payload in _jpeg_segments(blob, v.errors):
        if 0xE0 <= marker <= 0xEF:
            tag, _, _ = payload.partition(b"\x00")
            apps.append(f"APP{marker - 0xE0}/{tag[:12].decode('latin-1', 'replace')}")
        if marker == 0xFE:
            # The COM segment is where command-line pipelines park whatever
            # they were told to; A1111 and several wrappers dump the whole
            # parameter string here when writing JPEG.
            comment = payload.decode("utf-8", "replace").strip("\x00").strip()
            if comment:
                v.raw["jpeg:comment"] = _clip(comment)
                _scan_freetext(v, "JPEG:comment", comment, "JPEG comment segment")
                _scan_dst(v, "JPEG:comment", comment)
        elif marker == 0xE1:
            if payload.startswith(b"Exif\x00\x00"):
                exif = exif or payload
            elif payload.startswith(_XMP_APP1):
                xmp.append(payload[len(_XMP_APP1):].decode("utf-8", "replace"))
            elif payload.startswith(_XMP_APP1_EXT):
                # 32-byte GUID + 4-byte total length + 4-byte offset, then the chunk.
                xmp.append(payload[len(_XMP_APP1_EXT) + 40:].decode("utf-8", "replace"))
        elif marker == 0xEB and payload[:2] == b"JP" and len(payload) >= 8:
            # APP11 fragments one JUMBF box across segments: 2-byte box
            # instance, 4-byte packet sequence, then LBox+TBox repeated in
            # every packet. Keeping the repeats splices 8 junk bytes into the
            # box at each segment boundary, which desynchronises the walk --
            # and real manifests always split, because they carry a thumbnail.
            packet = struct.unpack(">I", payload[4:8])[0]
            c2pa += payload[8:] if packet <= 1 else payload[16:]
        elif marker == 0xEE and payload.startswith(b"Adobe"):
            apps.append("APP14/Adobe")
    return xmp, (bytes(c2pa) or None), apps, exif


def _scan_webp(blob: _Buffer) -> tuple[list[str], bytes | None, bytes | None]:
    xmp: list[str] = []
    c2pa: bytes | None = None
    exif: bytes | None = None
    for fourcc, data in _riff_chunks(blob):
        if fourcc == b"XMP ":
            xmp.append(data.decode("utf-8", "replace"))
        elif fourcc == b"EXIF":
            exif = exif or data
        elif fourcc == b"C2PA":
            c2pa = data
    return xmp, c2pa, exif


def _scan_bmff(blob: _Buffer) -> bytes | None:
    """The first JUMBF box that is a C2PA manifest, not merely the first one.

    'jumb' is JUMBF's own box type and other profiles use it too, so returning
    the first one would let an unrelated box hide a real manifest behind it.
    'uuid' is generic, so it has to name C2PA before it counts at all.
    """
    fallback: bytes | None = None
    for btype, data in _bmff_boxes(blob):
        if btype == b"jumb":
            if any(tag in _JUMBF_MANIFEST_TAGS for tag, _ in _jumbf_manifest(bytes(data))):
                return bytes(data)
        elif btype == b"uuid" and b"c2pa" in data[:512] and fallback is None:
            fallback = bytes(data)
    return fallback


def _exif_of(v: MetadataVerdict, img: Image.Image | None,
             raw_exif: bytes | None) -> Image.Exif | None:
    """The EXIF block, preferring the one our own container walk produced.

    For PNG that is the only affordable source: `PngImageFile.getexif()` calls
    `load()` because an eXIf chunk may follow IDAT, so asking Pillow costs a
    full pixel decode -- 80 ms on a 4 MPix file against 0.1 ms for parsing the
    chunk we already walked past. For a file Pillow refused to open it is the
    only source at all, which is how a truncated JPEG still yields its EXIF.
    """
    if img is None or img.format == "PNG":
        if raw_exif is None:
            return None
        exif = Image.Exif()
        try:
            exif.load(raw_exif)
        except Exception as exc:  # noqa: BLE001 - malformed EXIF is common and non-fatal
            v.errors.append(f"EXIF block unreadable: {exc}")
            return None
        return exif
    try:
        return img.getexif()
    except Exception as exc:  # noqa: BLE001
        v.errors.append(f"EXIF unreadable: {exc}")
        return None


def _scan_exif(v: MetadataVerdict, img: Image.Image | None, raw_exif: bytes | None = None) -> None:
    exif = _exif_of(v, img, raw_exif)
    if not exif:
        return
    try:
        sub = dict(exif.get_ifd(_EXIF_IFD))
    except Exception:  # noqa: BLE001
        sub = {}
    try:
        gps = dict(exif.get_ifd(_GPS_IFD))
    except Exception:  # noqa: BLE001
        gps = {}

    for tag, name in _TEXT_TAGS_IFD0.items():
        text = _exif_str(exif.get(tag), tag)
        if not text:
            continue
        v.raw[f"exif:{name}"] = _clip(text)
        if name == "Software" and not _is_a1111(text):
            _scan_names(v, f"EXIF:{name}", text, _ROLE_TOOL)
        else:
            _scan_freetext(v, f"EXIF:{name}", text, f"EXIF:{name}")
        _scan_dst(v, f"EXIF:{name}", text)

    for tag, name in _XP_TAGS.items():
        text = _exif_str(exif.get(tag), tag)
        if text:
            v.raw[f"exif:{name}"] = _clip(text)
            _scan_names(v, f"EXIF:{name}", text, _ROLE_FREETEXT)

    # UserComment lives in the Exif sub-IFD, which is where A1111 writes the
    # entire parameter string for JPEG and WebP output.
    comment = _exif_str(sub.get(0x9286) or exif.get(0x9286), 0x9286)
    if comment:
        v.raw["exif:UserComment"] = _clip(comment)
        _scan_freetext(v, "EXIF:UserComment", comment, "EXIF:UserComment", _ROLE_PARAMS)
        _scan_dst(v, "EXIF:UserComment", comment)

    present: list[str] = []
    for tag, name in _CAMERA_TAGS_IFD0.items():
        raw_value = exif.get(tag)
        value = _exif_str(raw_value, tag) if isinstance(raw_value, (str, bytes)) else raw_value
        if value not in (None, "", 0) and str(value).strip().lower() not in ("unknown", "none"):
            present.append(name)
            v.raw[f"exif:{name}"] = _clip(str(value), 80)
    for tag, name in _CAMERA_TAGS_EXIF.items():
        if sub.get(tag) not in (None, "", 0):
            present.append(name)
    if gps:
        present.append("GPS")
    v.camera_evidence.extend(present)


def _scan_iptc_iim(v: MetadataVerdict, img: Image.Image | None) -> None:
    """IPTC IIM from the Photoshop resource block Pillow already parsed."""
    if img is None:
        return
    block = (img.info or {}).get("photoshop", {}).get(0x0404)
    if not isinstance(block, bytes):
        return
    pos, n = 0, len(block)
    while pos + 5 <= n:
        if block[pos] != 0x1C:
            break
        record, dataset = block[pos + 1], block[pos + 2]
        size = struct.unpack(">H", block[pos + 3:pos + 5])[0]
        pos += 5
        if size & 0x8000:  # extended dataset: the low bits give the length's width
            width = size & 0x7FFF
            if pos + width > n:
                break
            size = int.from_bytes(block[pos:pos + width], "big")
            pos += width
        if pos + size > n:
            v.errors.append("IPTC IIM dataset runs past end of block")
            break
        value = block[pos:pos + size]
        pos += size
        name = _IIM_FIELDS.get(dataset) if record == 2 else None
        if not name:
            continue
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            text = value.decode("latin-1", "replace")
        text = text.strip()
        if not text:
            continue
        v.raw[f"iptc:{name}"] = _clip(text, 120)
        role = _ROLE_TOOL if name in _IIM_TOOL_FIELDS else _ROLE_FREETEXT
        _scan_names(v, f"IPTC:{name}", text, role)
        _scan_dst(v, f"IPTC:{name}", text)


def _is_aigc(prefix: str, local: str) -> bool:
    return bool(_AIGC_NAME.search(prefix) or _AIGC_NAME.search(local))


def _scan_xmp(v: MetadataVerdict, packets: list[str]) -> None:
    if not packets:
        return
    joined = "\n".join(packets)
    v.raw["xmp"] = _clip(joined, _MAX_XMP)
    _scan_dst(v, "XMP", joined)

    for local, role in (("CreatorTool", _ROLE_TOOL), ("Software", _ROLE_TOOL),
                        ("softwareAgent", _ROLE_TOOL), ("Credit", _ROLE_FREETEXT),
                        ("creator", _ROLE_FREETEXT), ("description", _ROLE_FREETEXT),
                        ("title", _ROLE_FREETEXT), ("rights", _ROLE_FREETEXT)):
        for value in _xmp_values(joined, local):
            v.raw.setdefault(f"xmp:{local}", _clip(value, 120))
            _scan_names(v, f"XMP:{local}", value, role)

    actions = _xmp_values(joined, "action")
    if actions:
        v.raw["xmp:history"] = _clip(", ".join(actions), 200)

    # Capture settings survive in XMP where EXIF does not: COCO's pipeline
    # stripped the EXIF IFD from all 5000 files but left tiff:/exif: properties
    # in the XMP packet on 7% of them, which is all the camera evidence this
    # corpus has.
    for local, name in (("Make", "Make"), ("Model", "Model"), ("Lens", "LensModel"),
                        ("ExposureTime", "ExposureTime"), ("FNumber", "FNumber"),
                        ("ISOSpeedRatings", "ISOSpeedRatings"), ("FocalLength", "FocalLength"),
                        ("DateTimeOriginal", "DateTimeOriginal"), ("GPSLatitude", "GPS")):
        if _xmp_values(joined, local):
            v.camera_evidence.append(name)
            v.raw.setdefault("xmp:camera", []).append(local)

    # A tag or attribute in a generative-AI namespace is itself the claim; the
    # value is usually just "1" or a platform id. Two things are not claims,
    # though: an xmlns declaration, which only says the vocabulary is in scope,
    # and a property that explicitly denies generation.
    flagged: set[str] = set()
    for prefix, local, value in re.findall(r"<\s*([\w.-]+):([\w.-]+)[^>]*>([^<]{0,64})", joined):
        if _is_aigc(prefix, local) and value.strip().lower() not in _DENIALS:
            flagged.add(f"{prefix}:{local}")
    for prefix, local, value in re.findall(r'[\s"]([\w.-]+):([\w.-]+)\s*=\s*"([^"]*)"', joined):
        if (prefix.lower() != "xmlns" and _is_aigc(prefix, local)
                and value.strip().lower() not in _DENIALS):
            flagged.add(f"{prefix}:{local}")
    if flagged:
        _note_ai(v, "XMP carries generative-AI namespace tags: "
                    f"{', '.join(sorted(flagged)[:4])}")
        _note_generator(v, "unknown (AIGC-labelled)")


def _jumbf_boxes(data: bytes, start: int = 0, end: int | None = None, depth: int = 0):
    """Yield (tag, label, payload) for every JUMBF superbox, depth first.

    JUMBF nests exactly like ISO-BMFF but states each box's identity in a
    leading 'jumd' description box: a 16-byte content-type UUID whose first
    four bytes are an ASCII tag, one toggle byte, then a NUL-terminated label.
    """
    pos = start
    end = len(data) if end is None else end
    for _ in range(_MAX_BOXES):
        if pos + 8 > end:
            return
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        btype = data[pos + 4:pos + 8]
        body = pos + 8
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            body, header = pos + 16, 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        if btype == b"jumb":
            tag, label, children = _jumbf_describe(data, body, pos + size)
            if tag is not None:
                yield tag, label, data[children:pos + size]
                if depth < 6:
                    yield from _jumbf_boxes(data, children, pos + size, depth + 1)
        pos += size


def _jumbf_describe(data: bytes, start: int, end: int) -> tuple[bytes | None, str, int]:
    """Read the 'jumd' box that must open a superbox: (tag, label, offset after it)."""
    if start + 8 > end or data[start + 4:start + 8] != b"jumd":
        return None, "", start
    size = struct.unpack(">I", data[start:start + 4])[0]
    if size < 25 or start + size > end:
        return None, "", start
    uuid = bytes(data[start + 8:start + 24])
    if uuid[4:] != _JUMBF_UUID_SUFFIX:
        return None, "", start
    label = ""
    if data[start + 24] & 0x02:  # toggle bit 1: a requestable label follows
        raw = bytes(data[start + 25:start + size])
        label = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return uuid[:4], label, start + size


def _c2pa_tokens(payload: bytes) -> set[str]:
    """The c2pa.* labels in a manifest, split against the known vocabulary.

    CBOR stores text strings back to back behind a one-byte length header, and
    that header is frequently an ASCII letter (0x6b is 'k'), so two adjacent
    labels read as one word to a plain regex: 'c2pa.actionskc2pa.created'.
    """
    found: set[str] = set()
    for match in _C2PA_TOKEN.findall(payload):
        run = match.decode("ascii")
        while run.startswith("c2pa."):
            for term in _C2PA_VOCAB:
                if run.startswith(term):
                    found.add(term)
                    run = run[len(term):]
                    break
            else:
                found.add(run.rstrip(".-")[:48])
                break
            nxt = run.find("c2pa.")
            if nxt < 0:
                break
            run = run[nxt:]
    # The digitalSourceType URI contains the authority, which is not a label.
    return found - {"c2pa.org"}


def _jumbf_manifest(payload: bytes) -> list[tuple[bytes, str]]:
    """(tag, label) for the JUMBF boxes in `payload`, however it was framed.

    PNG, WebP and JPEG hand over a complete 'jumb' box; a BMFF walk hands over
    that box's *contents*, which start at the 'jumd' description instead.
    """
    boxes = [(tag, label) for tag, label, _ in _jumbf_boxes(payload)]
    if boxes:
        return boxes
    tag, label, children = _jumbf_describe(payload, 0, len(payload))
    if tag is None:
        return []
    return [(tag, label)] + [(t, lbl) for t, lbl, _ in _jumbf_boxes(payload, children)]


def _scan_c2pa(v: MetadataVerdict, payload: bytes, container: str) -> None:
    boxes = _jumbf_manifest(payload)
    confirmed = any(tag in _JUMBF_MANIFEST_TAGS for tag, _ in boxes)
    # The string test is a backstop for a manifest this walk cannot parse, not
    # a detector: every payload reaching here came from a slot the C2PA spec
    # reserves, and nothing below turns presence alone into AI evidence.
    if not confirmed and b"c2pa" not in payload[:512]:
        # The slot belongs to C2PA but the content is not a manifest. Saying so
        # beats either silently ignoring it or claiming provenance we cannot see.
        v.errors.append(f"{container} holds no recognisable JUMBF manifest")
        return
    v.has_c2pa = True
    ev = C2paEvidence(container=container, manifest_bytes=len(payload),
                      structurally_confirmed=confirmed)
    tokens = _c2pa_tokens(payload)
    ev.actions = sorted(t for t in tokens if t in _C2PA_ACTIONS)
    ev.labels = sorted({t for t in tokens if t not in _C2PA_ACTIONS}
                       | {lbl for _, lbl in boxes if lbl})
    ev.claim_generator = (_cbor_text_after(payload, b"claim_generator")
                          or _cbor_text_after(payload, b"name",
                                              payload.find(b"claim_generator_info")))
    # Every term, not the first: a manifest carries its ingredients' source
    # types too, and an ingredient's digitalCapture sitting ahead of the active
    # claim's trainedAlgorithmicMedia would otherwise be the one we report.
    text = payload.decode("latin-1")
    terms = [m.group(1) for m in _DST_URI.finditer(text)]
    ev.digital_source_type = next((t for t in terms if t.lower() in _DST_AI), None) \
        or (terms[0] if terms else None)
    v.c2pa = ev
    v.raw["c2pa"] = {"container": container, "bytes": len(payload),
                     "claim_generator": ev.claim_generator, "actions": ev.actions,
                     "jumbf_parsed": confirmed, "validated": False}
    # Presence is not a verdict in either direction: cameras sign with C2PA too.
    if ev.claim_generator:
        _scan_names(v, "C2PA claim_generator", ev.claim_generator, _ROLE_TOOL)
    for term in terms:
        _scan_dst(v, "C2PA manifest", f"http://c2pa.org/digitalsourcetype/{term}")


def _cbor_text_after(data: bytes, key: bytes, start: int = 0) -> str | None:
    """The CBOR text string stored directly after `key`.

    Decoding the manifest properly needs a CBOR library and, for a real
    verdict, the whole C2PA trust stack. A map key is stored as a plain UTF-8
    run, though, so the value header sits right behind it and major type 3 is
    three lines to unpack.
    """
    if start < 0:
        return None
    idx = data.find(key, start)
    if idx < 0:
        return None
    pos = idx + len(key)
    if pos >= len(data):
        return None
    head = data[pos]
    if 0x60 <= head <= 0x77:
        size, pos = head - 0x60, pos + 1
    elif head == 0x78 and pos + 1 < len(data):
        size, pos = data[pos + 1], pos + 2
    elif head == 0x79 and pos + 2 < len(data):
        size, pos = struct.unpack(">H", data[pos + 1:pos + 3])[0], pos + 3
    else:
        return None
    if not 0 < size <= 512 or pos + size > len(data):
        return None
    return data[pos:pos + size].decode("utf-8", "replace").strip() or None


# -- JPEG container statistics --------------------------------------------

def _ijg_scaled(base: tuple[int, ...], quality: int) -> list[int]:
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return [min(255, max(1, (b * scale + 50) // 100)) for b in base]


def _quality_from_tables(luma: list[int], chroma: list[int]) -> tuple[int | None, int | None]:
    """Estimate the libjpeg quality, and say when the tables match it exactly.

    libjpeg derives all 64 entries from one scale factor, so inverting the
    median entry lands within a step of the answer; searching all 100 settings
    instead costs 15 ms a file, which is more than the rest of this module.
    Saturated entries carry no information about the scale, so they are
    dropped before taking the median.
    """
    if all(t == 1 for t in luma):
        # libjpeg scales every entry to 1 at quality 100, so the filter below
        # would drop the whole table: the clearest default table of all reads
        # as no table at all.
        return 100, (100 if len(chroma) != 64 or all(t == 1 for t in chroma) else None)
    scales = sorted((100 * t - 50) / b for t, b in zip(luma, _IJG_LUMA) if 1 < t < 255)
    if not scales:
        return None, None
    s = scales[len(scales) // 2]
    guess = 5000 / s if s > 100 else (200 - s) / 2
    best_q, best_err = None, None
    for q in {min(100, max(1, int(round(guess)) + d)) for d in (-1, 0, 1)}:
        err = sum(abs(a - b) for a, b in zip(luma, _ijg_scaled(_IJG_LUMA, q)))
        if len(chroma) == 64:
            err += sum(abs(a - b) for a, b in zip(chroma, _ijg_scaled(_IJG_CHROMA, q)))
        if best_err is None or err < best_err:
            best_q, best_err = q, err
    return best_q, (best_q if best_err == 0 else None)


def _jpeg_structure(img: Image.Image, apps: list[str]) -> JpegStructure:
    st = JpegStructure(app_segments=sorted(set(apps)),
                       progressive=bool(img.info.get("progressive")),
                       has_jfif="jfif" in img.info,
                       has_adobe=any(a.startswith("APP14") for a in apps))
    tables = getattr(img, "quantization", None) or {}
    st.n_quant_tables = len(tables)
    luma = list(tables.get(0, ()))
    if len(luma) == 64:
        chroma = list(tables.get(1, ()))
        st.estimated_quality, st.ijg_quality = _quality_from_tables(luma, chroma)
    try:
        st.subsampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(
            JpegImagePlugin.get_sampling(img))
    except Exception:  # noqa: BLE001 - grayscale and exotic layouts
        st.subsampling = None
    return st


@contextmanager
def _buffer(source: str | os.PathLike | bytes):
    """The file's bytes, mapped rather than copied when we are given a path.

    Metadata sits in a few kilobytes at the head and tail of the container, but
    the walk still has to step over everything in between. Mapping makes that
    cost the page faults for the chunk headers it lands on instead of a copy of
    the whole file: 2.13 -> 1.07 ms per file on DIV2K's 4.4 MB PNGs, a wash on
    small ones. It also bounds the memory a hostile file can demand, because
    the pixel data never becomes resident.
    """
    if not isinstance(source, (str, os.PathLike)):
        yield bytes(source), None
        return
    path = Path(source)
    with path.open("rb") as fh:
        try:
            mapped = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            # An empty file cannot be mapped, and neither can some virtual
            # filesystems; a plain read is correct and the file is small.
            yield fh.read(), path
            return
        try:
            yield mapped, path
        finally:
            mapped.close()


def read_metadata(source: str | os.PathLike | bytes) -> MetadataVerdict:
    """Extract provenance evidence from a file's metadata.

    Never raises for a readable file that merely lacks metadata, and never
    reports evidence *against* generation: an empty verdict means "nothing
    found", which is the common case because platforms strip metadata.
    """
    v = MetadataVerdict()
    with _buffer(source) as (blob, path):
        fmt = _sniff_format(blob)
        img: Image.Image | None = None
        try:
            # Header-only: nothing in this module needs pixels, and decoding a
            # 4 MPix PNG costs a thousand times what reading its chunks does.
            img = Image.open(path if path is not None else io.BytesIO(blob))
            fmt = img.format or fmt
        except Exception as exc:  # noqa: BLE001 - Pillow raises many types
            v.errors.append(f"Pillow cannot open the file ({exc}); metadata read from raw bytes")
        try:
            _scan_containers(v, blob, fmt, img)
        finally:
            if img is not None:
                img.close()

    v.raw["format"] = fmt
    v.ai_evidence = sorted(set(v.ai_evidence))
    v.weak_ai_evidence = sorted(set(v.weak_ai_evidence))
    v.camera_evidence = sorted(set(v.camera_evidence))
    return v


def _scan_containers(v: MetadataVerdict, blob: _Buffer, fmt: str,
                     img: Image.Image | None) -> None:
    xmp: list[str] = []
    c2pa: bytes | None = None
    raw_exif: bytes | None = None
    apps: list[str] = []
    if fmt == "PNG":
        xmp, c2pa, raw_exif = _scan_png(v, blob)
    elif fmt in ("JPEG", "MPO"):
        xmp, c2pa, apps, raw_exif = _scan_jpeg(v, blob)
    elif fmt == "WEBP":
        xmp, c2pa, raw_exif = _scan_webp(blob)
    elif fmt in ("BMFF", "AVIF", "HEIF", "HEIC"):
        c2pa = _scan_bmff(blob)

    if img is not None:
        embedded = (img.info or {}).get("XML:com.adobe.xmp") or (img.info or {}).get("xmp")
        if isinstance(embedded, bytes):
            embedded = embedded.decode("utf-8", "replace")
        if embedded and not any(embedded[:200] in packet for packet in xmp):
            xmp.append(str(embedded))
    if not xmp and fmt not in ("PNG", "WEBP"):
        # Last resort for containers we do not walk (TIFF, raw, odd writers)
        # and for JPEGs that park a packet after EOI. Skipped for PNG and WebP,
        # which have exactly one legal slot for XMP and we just read it -- the
        # scan is linear in file size and those files are the large ones.
        # A packet straddling the boundary of a file larger than 2 * the
        # window is missed; no writer puts one in the middle of a 16 MB file.
        n = len(blob)
        spans = ([(0, n)] if n <= 2 * _XMP_SCAN_WINDOW
                 else [(0, _XMP_SCAN_WINDOW), (n - _XMP_SCAN_WINDOW, n)])
        xmp = [m.group(0).decode("utf-8", "replace")
               for lo, hi in spans for m in _XMP_PACKET.finditer(blob, lo, hi)]

    _scan_exif(v, img, raw_exif)
    _scan_iptc_iim(v, img)
    _scan_xmp(v, xmp)
    if c2pa:
        _scan_c2pa(v, c2pa, {"PNG": "PNG caBX chunk", "JPEG": "JPEG APP11 JUMBF",
                             "WEBP": "WebP C2PA chunk"}.get(fmt, "ISO-BMFF jumb box"))

    if fmt in ("JPEG", "MPO") and img is not None:
        v.jpeg = _jpeg_structure(img, apps)
        v.raw["jpeg"] = {"quality": v.jpeg.estimated_quality,
                         "ijg_default_tables": v.jpeg.library_default_tables,
                         "subsampling": v.jpeg.subsampling,
                         "progressive": v.jpeg.progressive}
