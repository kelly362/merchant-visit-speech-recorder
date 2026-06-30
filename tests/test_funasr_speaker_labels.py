import unittest

import numpy as np

import funasr_server


def tone(frequency, seconds=1.0, sample_rate=16000):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class FunAsrSpeakerLabelsTest(unittest.TestCase):
    def test_uses_model_speaker_labels_when_available(self):
        result = [
            {
                "sentence_info": [
                    {"sentence": "您好我们看一下活动方案", "spk": 0},
                    {"sentence": "我想知道费用是多少", "spk": 1},
                ]
            }
        ]

        self.assertEqual(
            funasr_server.sentence_lines(result),
            [
                "说话人1：您好我们看一下活动方案。",
                "说话人2：我想知道费用是多少？",
            ],
        )

    def test_clusters_sentence_audio_when_model_returns_one_speaker(self):
        sample_rate = funasr_server.SAMPLE_RATE
        silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
        samples = np.concatenate(
            [
                tone(220),
                silence,
                tone(660),
                silence,
                tone(220),
            ]
        )
        result = [
            {
                "sentence_info": [
                    {
                        "sentence": "您好我们看一下活动方案",
                        "spk": 0,
                        "timestamp": [[0, 1000]],
                    },
                    {
                        "sentence": "我想知道费用是多少",
                        "spk": 0,
                        "timestamp": [[1200, 2200]],
                    },
                    {
                        "sentence": "这个可以从试点开始",
                        "spk": 0,
                        "timestamp": [[2400, 3400]],
                    },
                ]
            }
        ]

        self.assertEqual(
            funasr_server.sentence_lines(result, samples=samples),
            [
                "说话人1：您好我们看一下活动方案。",
                "说话人2：我想知道费用是多少？",
                "说话人1：这个可以从试点开始。",
            ],
        )

    def test_builds_sentences_from_word_timestamps_before_clustering(self):
        sample_rate = funasr_server.SAMPLE_RATE
        silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
        samples = np.concatenate(
            [
                tone(220),
                silence,
                tone(660),
                silence,
                tone(220),
            ]
        )
        result = [
            {
                "text": "您好费用多少可以试点",
                "words": list("您好费用多少可以试点"),
                "timestamp": [
                    [0, 120],
                    [180, 300],
                    [1200, 1320],
                    [1380, 1500],
                    [1560, 1680],
                    [1740, 1860],
                    [2400, 2520],
                    [2580, 2700],
                    [2760, 2880],
                    [2940, 3060],
                ],
            }
        ]

        self.assertEqual(
            funasr_server.sentence_lines(result, samples=samples),
            [
                "说话人1：您好。",
                "说话人2：费用多少？",
                "说话人1：可以试点。",
            ],
        )


if __name__ == "__main__":
    unittest.main()
