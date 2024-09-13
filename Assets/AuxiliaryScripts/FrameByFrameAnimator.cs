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
        Debug.Log(subframeRate);
    }

    void Start()
    {
        animator.speed = 0;
    }

    void Update()
    {
        numCurrSubframes++;
        animator.PlayInFixedTime(animationName, 0, numCurrSubframes / subframeRate);
        animator.Update(0);
    }
}