using UnityEngine;
using RosSharp.RosBridgeClient;

public class FrameByFrameAnimation : MonoBehaviour
{
    public Animation animation;
    public string animationName;
    private int numCurrSubframes = 0; // The current frame time

    public GameObject radar;
    private ScreenSpaceRadarControlPlot ssrcp;
    private float subframeRate; // The frame rate of the radar

    void Awake()
    {
        ssrcp = radar.GetComponent<ScreenSpaceRadarControlPlot>();
        subframeRate = ssrcp.GetSubframeRate();
        UnityEngine.Debug.Log(subframeRate);
        subframeRate = 600.0f;
    }

    void Start()
    {
        animation[animationName].speed = 0; // Stop automatic playback
    }

    void Update()
    {
        numCurrSubframes++;
        animation[animationName].time = numCurrSubframes / subframeRate;
        animation.Sample(); // Update the animation
    }
}