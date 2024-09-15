using UnityEngine;
using RosSharp.RosBridgeClient;

public class FrameByFrameAnimator : MonoBehaviour
{
    public Animator animator;
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
        animator.speed = 0; // Stop automatic playback
    }

    void Update()
    {
        numCurrSubframes++;
        animator.Play(animationName, 0, numCurrSubframes / subframeRate); // Set the animation to the current frame
        animator.Update(1 / subframeRate); // Update the animation
    }
}