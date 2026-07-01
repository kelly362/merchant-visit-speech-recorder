import unittest

import xfyun_rtasr_server as relay


class XfyunRelayTest(unittest.TestCase):
    def test_punctuates_plain_statement(self):
        self.assertEqual(relay.ensure_punctuation("今天先测试实时转写"), "今天先测试实时转写。")

    def test_punctuates_question(self):
        self.assertEqual(relay.ensure_punctuation("这个费用是多少"), "这个费用是多少？")

    def test_strips_leading_fragment_punctuation(self):
        self.assertEqual(relay.ensure_punctuation("，而且人类也不一定希望这些东西"), "而且人类也不一定希望这些东西。")

    def test_normalizes_zero_based_speaker(self):
        self.assertEqual(relay.normalize_speaker("0"), "说话人1")
        self.assertEqual(relay.normalize_speaker("1"), "说话人2")

    def test_parses_direct_segment_payload(self):
        raw = {
            "speaker": "1",
            "text": "这个费用是多少",
            "is_final": True,
        }

        self.assertEqual(
            relay.parse_xfyun_message(raw),
            [
                {
                    "type": "result",
                    "speaker": "说话人2",
                    "text": "这个费用是多少？",
                    "is_final": True,
                }
            ],
        )

    def test_parses_cn_st_words_by_role(self):
        raw = {
            "code": "0",
            "data": {
                "cn": {
                    "st": {
                        "type": "0",
                        "rt": [
                            {
                                "ws": [
                                    {"cw": [{"w": "你好", "rl": "0"}]},
                                    {"cw": [{"w": "我们看一下方案", "rl": "0"}]},
                                    {"cw": [{"w": "费用是多少", "rl": "1"}]},
                                ]
                            }
                        ],
                    }
                }
            },
        }

        self.assertEqual(
            relay.parse_xfyun_message(raw),
            [
                {
                    "type": "result",
                    "speaker": "说话人1",
                    "text": "你好我们看一下方案。",
                    "is_final": True,
                },
                {
                    "type": "result",
                    "speaker": "说话人2",
                    "text": "费用是多少？",
                    "is_final": True,
                },
            ],
        )

    def test_transcript_lines_to_segments(self):
        transcript = "说话人1：你好我们开始。\n说话人2：可以。"

        self.assertEqual(
            relay.transcript_lines_to_segments(transcript),
            [
                {"speaker": "说话人1", "text": "你好我们开始。"},
                {"speaker": "说话人2", "text": "可以。"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
