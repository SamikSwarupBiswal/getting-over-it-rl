using System;
using System.IO;
using UnityEngine;

// A read-only spectator HUD. JSON comes from scripts/watch_training.py and
// never enters the learner's observations, rewards or checkpoint state.
sealed class TrainingHud : MonoBehaviour
{
    public string DataPath;
    // Unity's JSON deserializer assigns these fields through reflection.
#pragma warning disable 0649
    [Serializable]
    sealed class Frame
    {
        public int update, policyUpdate, parallelGames, episode;
        public int distanceAttempt, fastestAttempt;
        public float recordDistance, fastestSeconds, targetProgress, targetSeconds;
        public string lastReset;
        public long trainingActions;
        public float actionsPerSecond, elapsed, previewProgress, previewBest, previewReturn, modelAge, previewSeconds;
        public string phase, error, trainingStatus;
        public float[] inputValues, hidden1, hidden2, outputs, edges1, edges2, edges3;
        public float[] progressTimes, progressHistory;
    }
#pragma warning restore 0649

    Frame data;
    float nextRead;
    DateTime lastStamp;
    string readError;
    Texture2D dot;
    GUIStyle text, title, tiny;

    void Update()
    {
        if (string.IsNullOrEmpty(DataPath) || Time.unscaledTime < nextRead) return;
        nextRead = Time.unscaledTime + 0.1f;
        try
        {
            var stamp = File.GetLastWriteTimeUtc(DataPath);
            if (stamp == lastStamp) return;
            using (var stream = new FileStream(DataPath, FileMode.Open, FileAccess.Read,
                                               FileShare.ReadWrite | FileShare.Delete))
            using (var reader = new StreamReader(stream))
                data = JsonUtility.FromJson<Frame>(reader.ReadToEnd());
            lastStamp = stamp;
            readError = null;
        }
        catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException || ex is ArgumentException)
        {
            readError = "Preview telemetry unavailable";
        }
    }

    void InitDrawing()
    {
        if (text != null) return;
        text = new GUIStyle(GUI.skin.label) { fontSize = 16 };
        text.normal.textColor = new Color(0.03f, 0.07f, 0.07f);
        title = new GUIStyle(text) { fontSize = 17, fontStyle = FontStyle.Bold };
        tiny = new GUIStyle(text) { fontSize = 11 };
        dot = new Texture2D(32, 32, TextureFormat.RGBA32, false);
        var pixels = new Color[32 * 32];
        for (var y = 0; y < 32; y++)
            for (var x = 0; x < 32; x++)
                pixels[y * 32 + x] = new Color(1, 1, 1,
                    Mathf.Clamp01(15.5f - Vector2.Distance(new Vector2(x, y), new Vector2(15.5f, 15.5f))));
        dot.SetPixels(pixels);
        dot.Apply();
    }

    void OnGUI()
    {
        if (Patches.Ending && Patches.Won && data == null) return;
        InitDrawing();
        var oldMatrix = GUI.matrix;
        var oldColor = GUI.color;
        float scale = Mathf.Clamp(Screen.width / 1600f, 0.65f, 1.5f);
        GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1));
        float width = Screen.width / scale, height = Screen.height / scale;
        try
        {
            Panel(new Rect(10, 10, width - 20, 73), 0.63f);
            GUI.Label(new Rect(22, 14, width - 40, 24), "GETTING OVER IT | GPU PPO | LIVE SHARED-POLICY PREVIEW", title);
            if (data == null)
            {
                GUI.Label(new Rect(22, 42, width - 40, 25), readError ?? "Waiting for the saved policy...", text);
                return;
            }
            GUI.Label(new Rect(22, 38, width - 40, 23),
                $"{data.parallelGames:N0} training games, one network | update {data.update:N0} | " +
                $"{data.trainingActions:N0} actions | {data.actionsPerSecond:N0} actions/sec | {data.elapsed / 60f:F1} min | {data.trainingStatus}", text);
            GUI.Label(new Rect(22, 59, width - 40, 22),
                $"Preview from start: route {data.previewProgress:F1}, best {data.previewBest:F1} | " +
                $"return {data.previewReturn:+0.00;-0.00;0.00} | attempt {data.episode} | weights update {data.policyUpdate:N0}, {data.modelAge:F1}s old" +
                (readError != null ? " | " + readError : "") + (!string.IsNullOrEmpty(data.error) ? " | " + data.error : ""), tiny);

            float top = height - 174;
            Panel(new Rect(10, top, Mathf.Min(880, width - 20), 164), 0.29f);
            GUI.Label(new Rect(22, top + 4, 830, 22), "LIVE POLICY | actual activations and sampled learned weights", text);
            int groups = data.inputValues != null && data.inputValues.Length == 7 ? 7 : 6;
            var input = Positions(104, top + 34, groups, 16);
            var h1 = Positions(290, top + 34, 16, 6.7f);
            var h2 = Positions(535, top + 34, 16, 6.7f);
            var output = Positions(754, top + 56, 2, 52);
            Edges(input, h1, data.edges1, 2);
            Edges(h1, h2, data.edges2, 3);
            Edges(h2, output, data.edges3, 6);
            string[] names = { "pot", "hammer", "mouse", "touch", "motors", "sight", "time" };
            for (var i = 0; i < input.Length; i++)
                GUI.Label(new Rect(22, input[i].y - 10, 75, 22), names[i], tiny);
            Nodes(input, data.inputValues, 4.5f);
            Nodes(h1, data.hidden1, 3);
            Nodes(h2, data.hidden2, 3);
            Nodes(output, data.outputs, 7);
            for (var i = 0; i < 2; i++)
                GUI.Label(new Rect(768, output[i].y - 11, 100, 24),
                    (i == 0 ? "X " : "Y ") + Value(data.outputs, i).ToString("+0.00;-0.00;0.00"), text);
            GUI.Label(new Rect(22, top + 140, 840, 20),
                "Input groups; 16 neurons shown per hidden layer. Green + / orange -. Preview uses the latest saved training policy.", tiny);
            DrawProgress(new Rect(width - 422, top - 27, 412, 191));
        }
        finally
        {
            GUI.matrix = oldMatrix;
            GUI.color = oldColor;
        }
    }

    static Vector2[] Positions(float x, float y, int count, float spacing)
    {
        var positions = new Vector2[count];
        for (var i = 0; i < count; i++) positions[i] = new Vector2(x, y + i * spacing);
        return positions;
    }

    void DrawProgress(Rect panel)
    {
        Panel(panel, 0.83f);
        GUI.Label(new Rect(panel.x + 12, panel.y + 3, 390, 22), $"ROUTE DISTANCE  /  ATTEMPT {data.episode}", text);
        GUI.Label(new Rect(panel.x + 12, panel.y + 25, 390, 19),
            $"Solid: live {data.previewProgress:F1}   |   Dashed: farthest {data.recordDistance:F1} (#{data.distanceAttempt})", tiny);
        string fastest = data.fastestSeconds > 0 ? $"{data.fastestSeconds:F2}s (#{data.fastestAttempt})" : "pending";
        if (data.targetProgress > 0)
        {
            float clock = data.targetSeconds > 0 ? data.targetSeconds : data.previewSeconds;
            GUI.Label(new Rect(panel.x + 12, panel.y + 42, 390, 19),
                $"To {data.targetProgress:F0} m: {clock:F2}s  |  Fastest: {fastest}", tiny);
        }
        var plot = new Rect(panel.x + 38, panel.y + 66, panel.width - 57, 76);
        float high = Mathf.Max(1, Mathf.Max(data.targetProgress, data.recordDistance)), low = 0;
        int n = data.progressHistory == null || data.progressTimes == null ? 0 :
            Mathf.Min(data.progressHistory.Length, data.progressTimes.Length);
        for (var i = 0; i < n; i++)
        {
            high = Mathf.Max(high, Value(data.progressHistory, i));
            low = Mathf.Min(low, Value(data.progressHistory, i));
        }
        var axis = new Color(0.1f, 0.25f, 0.23f, 0.6f);
        Line(new Vector2(plot.x, plot.y), new Vector2(plot.x, plot.yMax), axis);
        Line(new Vector2(plot.x, plot.yMax), new Vector2(plot.xMax, plot.yMax), axis);
        GUI.color = Color.white;
        GUI.Label(new Rect(panel.x + 5, plot.y - 10, 30, 22), high.ToString("F0"), tiny);
        GUI.Label(new Rect(panel.x + 5, plot.yMax - 10, 30, 22), low.ToString("F0"), tiny);
        float span = Mathf.Max(10, 10 * Mathf.Ceil(Mathf.Max(data.previewSeconds, data.fastestSeconds) / 10));
        float bestY = plot.yMax - (data.recordDistance - low) / (high - low) * plot.height;
        for (float x = plot.x; x < plot.xMax; x += 9)
            Line(new Vector2(x, bestY), new Vector2(Mathf.Min(x + 5, plot.xMax), bestY), new Color(0.62f, 0.30f, 0.04f));
        if (data.fastestSeconds > 0)
        {
            float x = plot.x + data.fastestSeconds / span * plot.width;
            for (float y = plot.y; y < plot.yMax; y += 6)
                Line(new Vector2(x, y), new Vector2(x, Mathf.Min(y + 2, plot.yMax)), new Color(0.43f, 0.19f, 0.56f));
        }
        Vector2 previous = default(Vector2);
        for (var i = 0; i < n; i++)
        {
            var point = new Vector2(plot.x + Mathf.Clamp01(Value(data.progressTimes, i) / span) * plot.width,
                plot.yMax - Mathf.Clamp01((Value(data.progressHistory, i) - low) / (high - low)) * plot.height);
            if (i > 0) Line(previous, point, new Color(0.02f, 0.28f, 0.49f));
            previous = point;
        }
        GUI.color = Color.white;
        GUI.Label(new Rect(plot.x - 4, plot.yMax, 70, 18), "0 s", tiny);
        GUI.Label(new Rect(plot.center.x - 46, plot.yMax, 110, 18), "Elapsed game time", tiny);
        GUI.Label(new Rect(plot.xMax - 25, plot.yMax, 48, 18), $"{span:F0} s", tiny);
        GUI.Label(new Rect(panel.x + 12, panel.y + 161, 390, 23),
            "Last reset: " + (string.IsNullOrEmpty(data.lastReset) ? "none recorded" : data.lastReset), tiny);
    }

    static float Value(float[] values, int i) => values != null && i < values.Length &&
        !float.IsNaN(values[i]) && !float.IsInfinity(values[i]) ? values[i] : 0;

    static void Panel(Rect rect, float opacity)
    {
        GUI.color = new Color(1, 1, 1, opacity);
        GUI.DrawTexture(rect, Texture2D.whiteTexture);
        GUI.color = Color.white;
    }

    void Nodes(Vector2[] positions, float[] values, float radius)
    {
        for (var i = 0; i < positions.Length; i++)
        {
            float v = Value(values, i);
            GUI.color = Color.white;
            GUI.DrawTexture(new Rect(positions[i].x - radius - 1, positions[i].y - radius - 1, 2 * radius + 2, 2 * radius + 2), dot);
            GUI.color = Color.Lerp(new Color(0.3f, 0.36f, 0.31f), v >= 0 ? new Color(0.55f, 0.85f, 0.13f) :
                new Color(1, 0.42f, 0.08f), Mathf.Clamp01(Mathf.Abs(v)));
            GUI.DrawTexture(new Rect(positions[i].x - radius, positions[i].y - radius, 2 * radius, 2 * radius), dot);
        }
        GUI.color = Color.white;
    }

    static void Edges(Vector2[] from, Vector2[] to, float[] weights, int strongest)
    {
        if (weights == null || weights.Length != from.Length * to.Length) return;
        float max = 0.00001f;
        foreach (var w in weights) if (!float.IsNaN(w) && !float.IsInfinity(w)) max = Mathf.Max(max, Mathf.Abs(w));
        for (var j = 0; j < to.Length; j++)
            for (var i = 0; i < from.Length; i++)
            {
                float w = Value(weights, i * to.Length + j);
                int larger = 0;
                for (var k = 0; k < from.Length; k++)
                    if (Mathf.Abs(Value(weights, k * to.Length + j)) > Mathf.Abs(w)) larger++;
                if (larger >= strongest || Mathf.Abs(w) < 0.00001f) continue;
                var color = w >= 0 ? new Color(0.42f, 0.7f, 0.06f) : new Color(1, 0.4f, 0.05f);
                color.a = 0.12f + 0.48f * Mathf.Abs(w) / max;
                Line(from[i], to[j], color);
            }
        GUI.color = Color.white;
    }

    static void Line(Vector2 a, Vector2 b, Color color)
    {
        var matrix = GUI.matrix;
        GUI.color = color;
        GUI.matrix = matrix * Matrix4x4.TRS(a,
            Quaternion.Euler(0, 0, Mathf.Atan2(b.y - a.y, b.x - a.x) * Mathf.Rad2Deg), Vector3.one);
        GUI.DrawTexture(new Rect(0, 0, Vector2.Distance(a, b), 0.8f), Texture2D.whiteTexture);
        GUI.matrix = matrix;
    }

    void OnDestroy()
    {
        if (dot != null) Destroy(dot);
    }
}
