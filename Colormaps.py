"""
self added the cloudnet velocity and segmented colormap
"""

import matplotlib

velocity_colors = (
    # (1.0, 1.0, 1.0),
    (0.0, 0.0, 0.3),  # end_val dark blue
    (0.0274509803922, 0.0549019607843, 0.23137254902),
    (0.0392156862745, 0.0549019607843, 0.250980392157),
    (0.0509803921569, 0.0549019607843, 0.266666666667),
    (0.0627450980392, 0.0549019607843, 0.286274509804),
    (0.0705882352941, 0.0549019607843, 0.298039215686),
    (0.0823529411765, 0.0549019607843, 0.313725490196),
    (0.0901960784314, 0.0549019607843, 0.333333333333),
    (0.0980392156863, 0.0549019607843, 0.349019607843),
    (0.109803921569, 0.0549019607843, 0.364705882353),
    (0.121568627451, 0.0549019607843, 0.38431372549),
    (0.133333333333, 0.0549019607843, 0.4),
    (0.141176470588, 0.0549019607843, 0.41568627451),
    (0.152941176471, 0.0549019607843, 0.43137254902),
    (0.160784313725, 0.0549019607843, 0.447058823529),
    (0.172549019608, 0.0549019607843, 0.466666666667),
    (0.18431372549, 0.0549019607843, 0.482352941176),
    (0.192156862745, 0.0549019607843, 0.498039215686),
    (0.203921568627, 0.0549019607843, 0.513725490196),
    (0.211764705882, 0.0549019607843, 0.529411764706),
    (0.223529411765, 0.0549019607843, 0.549019607843),
    (0.23137254902, 0.0549019607843, 0.564705882353),
    (0.243137254902, 0.0549019607843, 0.580392156863),
    (0.254901960784, 0.0549019607843, 0.596078431373),
    (0.262745098039, 0.0549019607843, 0.611764705882),
    (0.274509803922, 0.0549019607843, 0.627450980392),
    (0.282352941176, 0.0549019607843, 0.647058823529),
    (0.294117647059, 0.0549019607843, 0.662745098039),
    (0.305882352941, 0.0549019607843, 0.682352941176),
    (0.317647058824, 0.0549019607843, 0.698039215686),
    (0.325490196078, 0.0588235294118, 0.713725490196),
    (0.333333333333, 0.0666666666667, 0.729411764706),
    (0.345098039216, 0.105882352941, 0.745098039216),
    (0.352941176471, 0.145098039216, 0.760784313725),
    (0.364705882353, 0.18431372549, 0.780392156863),
    (0.376470588235, 0.219607843137, 0.796078431373),
    (0.388235294118, 0.258823529412, 0.811764705882),
    (0.396078431373, 0.298039215686, 0.827450980392),
    (0.403921568627, 0.337254901961, 0.843137254902),
    (0.41568627451, 0.372549019608, 0.862745098039),
    (0.427450980392, 0.411764705882, 0.878431372549),
    (0.439215686275, 0.450980392157, 0.894117647059),
    (0.447058823529, 0.486274509804, 0.909803921569),
    (0.458823529412, 0.525490196078, 0.925490196078),
    (0.466666666667, 0.564705882353, 0.945098039216),
    (0.474509803922, 0.603921568627, 0.960784313725),
    (0.486274509804, 0.639215686275, 0.976470588235),
    (0.498039215686, 0.678431372549, 0.996078431373),
    (0.482352941176, 0.678431372549, 0.956862745098),
    (0.450980392157, 0.662745098039, 0.898039215686),
    (0.419607843137, 0.647058823529, 0.835294117647),
    (0.388235294118, 0.63137254902, 0.772549019608),
    (0.356862745098, 0.61568627451, 0.713725490196),
    (0.329411764706, 0.6, 0.650980392157),
    (0.298039215686, 0.58431372549, 0.592156862745),
    (0.266666666667, 0.564705882353, 0.529411764706),
    (0.235294117647, 0.549019607843, 0.470588235294),
    (0.207843137255, 0.533333333333, 0.407843137255),
    (0.176470588235, 0.517647058824, 0.345098039216),
    (0.145098039216, 0.501960784314, 0.286274509804),
    (0.113725490196, 0.486274509804, 0.223529411765),
    (0.0823529411765, 0.470588235294, 0.164705882353),
    (0.0549019607843, 0.458823529412, 0.101960784314),
    (0.0235294117647, 0.450980392157, 0.043137254902),
    (0.00392156862745, 0.447058823529, 0.0),
    (0.0, 0.454901960784, 0.0),
    (0.0, 0.462745098039, 0.0),
    (0.0, 0.470588235294, 0.0),
    (0.0, 0.478431372549, 0.0),
    (0.0, 0.486274509804, 0.0),
    (0.0, 0.494117647059, 0.0),
    (0.0, 0.501960784314, 0.0),
    (0.0, 0.509803921569, 0.0),
    (0.0, 0.517647058824, 0.0),
    (0.0, 0.525490196078, 0.0),
    (0.0, 0.533333333333, 0.0),
    (0.0, 0.541176470588, 0.0),
    (0.0, 0.549019607843, 0.0),
    (0.0, 0.556862745098, 0.0),
    (0.0078431372549, 0.564705882353, 0.0078431372549),
    (0.0196078431373, 0.572549019608, 0.0196078431373),
    (0.0352941176471, 0.580392156863, 0.0352941176471),
    (0.0549019607843, 0.58431372549, 0.0549019607843),
    (0.078431372549, 0.588235294118, 0.078431372549),
    (0.109803921569, 0.588235294118, 0.109803921569),
    (0.129411764706, 0.6, 0.129411764706),
    (0.149019607843, 0.619607843137, 0.149019607843),
    (0.16862745098, 0.63137254902, 0.16862745098),
    (0.18431372549, 0.639215686275, 0.18431372549),
    (0.203921568627, 0.647058823529, 0.203921568627),
    (0.223529411765, 0.654901960784, 0.223529411765),
    (0.243137254902, 0.662745098039, 0.243137254902),
    (0.258823529412, 0.670588235294, 0.258823529412),
    (0.278431372549, 0.678431372549, 0.278431372549),
    (0.298039215686, 0.686274509804, 0.298039215686),
    (0.317647058824, 0.694117647059, 0.317647058824),
    (0.337254901961, 0.698039215686, 0.337254901961),
    (0.356862745098, 0.705882352941, 0.356862745098),
    (0.372549019608, 0.713725490196, 0.372549019608),
    (0.388235294118, 0.721568627451, 0.388235294118),
    (0.407843137255, 0.729411764706, 0.407843137255),
    (0.427450980392, 0.733333333333, 0.427450980392),
    (0.447058823529, 0.741176470588, 0.447058823529),
    (0.466666666667, 0.749019607843, 0.466666666667),
    (0.486274509804, 0.756862745098, 0.486274509804),
    (0.505882352941, 0.764705882353, 0.505882352941),
    (0.521568627451, 0.772549019608, 0.521568627451),
    (0.541176470588, 0.780392156863, 0.541176470588),
    (0.560784313725, 0.788235294118, 0.560784313725),
    (0.576470588235, 0.796078431373, 0.576470588235),
    (0.596078431373, 0.803921568627, 0.596078431373),
    (0.61568627451, 0.811764705882, 0.61568627451),
    (0.635294117647, 0.819607843137, 0.635294117647),
    (0.650980392157, 0.827450980392, 0.650980392157),
    (0.670588235294, 0.835294117647, 0.670588235294),
    (0.690196078431, 0.843137254902, 0.690196078431),
    (0.709803921569, 0.850980392157, 0.709803921569),
    (0.729411764706, 0.858823529412, 0.729411764706),
    (0.745098039216, 0.866666666667, 0.745098039216),
    (0.764705882353, 0.870588235294, 0.764705882353),
    (0.78431372549, 0.878431372549, 0.78431372549),
    (0.8, 0.886274509804, 0.8),                       # light green
    (0.819607843137, 0.894117647059, 0.819607843137),
    (0.839215686275, 0.901960784314, 0.839215686275),
    (0.858823529412, 0.909803921569, 0.858823529412),
    (0.878431372549, 0.917647058824, 0.878431372549),
    (0.83137254902, 0.858823529412, 0.83137254902),
    (0.737254901961, 0.745098039216, 0.737254901961),
    (0.752941176471, 0.749019607843, 0.737254901961),
    (0.874509803922, 0.866666666667, 0.83137254902),
    (0.941176470588, 0.929411764706, 0.858823529412), # brightest value
    (0.941176470588, 0.921568627451, 0.8),
    (0.941176470588, 0.913725490196, 0.741176470588),
    (0.941176470588, 0.905882352941, 0.686274509804),
    (0.941176470588, 0.898039215686, 0.627450980392),
    (0.941176470588, 0.890196078431, 0.572549019608),
    (0.941176470588, 0.882352941176, 0.513725490196),
    (0.941176470588, 0.878431372549, 0.454901960784),
    (0.941176470588, 0.870588235294, 0.4),            # light yellow
    (0.941176470588, 0.862745098039, 0.341176470588),
    (0.941176470588, 0.854901960784, 0.286274509804),
    (0.941176470588, 0.847058823529, 0.227450980392),
    (0.941176470588, 0.839215686275, 0.16862745098),
    (0.941176470588, 0.83137254902, 0.113725490196),
    (0.941176470588, 0.823529411765, 0.0588235294118),
    (0.941176470588, 0.81568627451, 0.0274509803922),
    (0.941176470588, 0.807843137255, 0.0),
    (0.941176470588, 0.8, 0.00392156862745),
    (0.941176470588, 0.792156862745, 0.0078431372549),
    (0.941176470588, 0.78431372549, 0.0117647058824),
    (0.941176470588, 0.776470588235, 0.0196078431373),
    (0.941176470588, 0.76862745098, 0.0235294117647),
    (0.941176470588, 0.760784313725, 0.0313725490196),
    (0.941176470588, 0.752941176471, 0.0352941176471),
    (0.941176470588, 0.745098039216, 0.0392156862745),
    (0.941176470588, 0.741176470588, 0.043137254902),
    (0.941176470588, 0.733333333333, 0.0509803921569),
    (0.941176470588, 0.725490196078, 0.0549019607843),
    (0.941176470588, 0.717647058824, 0.0588235294118),
    (0.941176470588, 0.709803921569, 0.0666666666667),
    (0.941176470588, 0.701960784314, 0.0705882352941),
    (0.941176470588, 0.694117647059, 0.0745098039216),
    (0.941176470588, 0.686274509804, 0.078431372549),
    (0.937254901961, 0.678431372549, 0.0862745098039),
    (0.925490196078, 0.670588235294, 0.0901960784314),
    (0.917647058824, 0.662745098039, 0.0941176470588),
    (0.905882352941, 0.654901960784, 0.0980392156863),
    (0.898039215686, 0.647058823529, 0.105882352941),
    (0.890196078431, 0.639215686275, 0.109803921569),
    (0.878431372549, 0.63137254902, 0.117647058824),
    (0.862745098039, 0.619607843137, 0.125490196078),
    (0.850980392157, 0.607843137255, 0.129411764706),
    (0.843137254902, 0.6, 0.137254901961),
    (0.83137254902, 0.592156862745, 0.141176470588),
    (0.827450980392, 0.588235294118, 0.145098039216),
    (0.819607843137, 0.580392156863, 0.149019607843),
    (0.807843137255, 0.572549019608, 0.156862745098),
    (0.8, 0.564705882353, 0.160784313725),
    (0.788235294118, 0.556862745098, 0.16862745098),
    (0.780392156863, 0.549019607843, 0.172549019608),
    (0.76862745098, 0.541176470588, 0.176470588235),
    (0.760784313725, 0.533333333333, 0.180392156863),
    (0.749019607843, 0.525490196078, 0.188235294118),
    (0.741176470588, 0.517647058824, 0.192156862745),
    (0.733333333333, 0.509803921569, 0.196078431373),
    (0.725490196078, 0.501960784314, 0.203921568627),
    (0.717647058824, 0.494117647059, 0.207843137255),
    (0.705882352941, 0.486274509804, 0.211764705882),
    (0.698039215686, 0.478431372549, 0.21568627451),
    (0.686274509804, 0.470588235294, 0.223529411765),
    (0.678431372549, 0.462745098039, 0.227450980392),
    (0.670588235294, 0.454901960784, 0.235294117647),
    (0.662745098039, 0.450980392157, 0.239215686275),
    (0.654901960784, 0.443137254902, 0.243137254902),
    (0.643137254902, 0.435294117647, 0.247058823529),
    (0.635294117647, 0.427450980392, 0.250980392157),
    (0.623529411765, 0.419607843137, 0.250980392157),
    (0.61568627451, 0.411764705882, 0.250980392157),
    (0.607843137255, 0.403921568627, 0.250980392157),
    (0.596078431373, 0.396078431373, 0.250980392157),
    (0.588235294118, 0.388235294118, 0.250980392157),
    (0.580392156863, 0.380392156863, 0.250980392157),
    (0.572549019608, 0.372549019608, 0.250980392157),
    (0.560784313725, 0.364705882353, 0.250980392157),
    (0.552941176471, 0.356862745098, 0.250980392157),
    (0.545098039216, 0.349019607843, 0.250980392157),
    (0.533333333333, 0.341176470588, 0.250980392157),
    (0.525490196078, 0.333333333333, 0.250980392157),
    (0.517647058824, 0.325490196078, 0.250980392157),
    (0.509803921569, 0.317647058824, 0.250980392157),
    (0.513725490196, 0.313725490196, 0.254901960784),
    (0.541176470588, 0.313725490196, 0.270588235294),
    (0.56862745098, 0.313725490196, 0.286274509804),
    (0.6, 0.313725490196, 0.301960784314),
    (0.63137254902, 0.313725490196, 0.317647058824),
    (0.662745098039, 0.313725490196, 0.329411764706),
    (0.694117647059, 0.313725490196, 0.345098039216),
    (0.721568627451, 0.313725490196, 0.360784313725),
    (0.752941176471, 0.313725490196, 0.376470588235),
    (0.780392156863, 0.313725490196, 0.392156862745),
    (0.811764705882, 0.313725490196, 0.407843137255),
    (0.839215686275, 0.313725490196, 0.423529411765),
    (0.870588235294, 0.313725490196, 0.439215686275),
    (0.901960784314, 0.313725490196, 0.454901960784),
    (0.933333333333, 0.313725490196, 0.466666666667),
    (0.964705882353, 0.313725490196, 0.482352941176),
    (0.992156862745, 0.313725490196, 0.498039215686),
    (0.992156862745, 0.305882352941, 0.490196078431),
    (0.980392156863, 0.298039215686, 0.474509803922),
    (0.96862745098, 0.286274509804, 0.462745098039),
    (0.956862745098, 0.278431372549, 0.447058823529),
    (0.945098039216, 0.266666666667, 0.435294117647),
    (0.933333333333, 0.254901960784, 0.419607843137),
    (0.921568627451, 0.243137254902, 0.407843137255),
    (0.909803921569, 0.235294117647, 0.392156862745),
    (0.898039215686, 0.227450980392, 0.380392156863),
    (0.886274509804, 0.21568627451, 0.364705882353),
    (0.878431372549, 0.207843137255, 0.349019607843),
    (0.866666666667, 0.196078431373, 0.337254901961),
    (0.854901960784, 0.18431372549, 0.321568627451),
    (0.843137254902, 0.172549019608, 0.309803921569),
    (0.83137254902, 0.164705882353, 0.298039215686),
    (0.81568627451, 0.156862745098, 0.282352941176),
    (0.803921568627, 0.145098039216, 0.270588235294),
    (0.792156862745, 0.133333333333, 0.254901960784),
    (0.780392156863, 0.121568627451, 0.239215686275),
    (0.76862745098, 0.113725490196, 0.227450980392),
    (0.756862745098, 0.101960784314, 0.211764705882),
    (0.745098039216, 0.0941176470588, 0.2),
    (0.737254901961, 0.0862745098039, 0.18431372549),
    (0.725490196078, 0.0745098039216, 0.172549019608),
    (0.713725490196, 0.0627450980392, 0.160784313725),
    (0.701960784314, 0.0509803921569, 0.145098039216),
    (0.690196078431, 0.043137254902, 0.133333333333),
    (0.678431372549, 0.0313725490196, 0.117647058824),
    (0.666666666667, 0.0235294117647, 0.101960784314)
)

carbonne_map = matplotlib.colors.ListedColormap(velocity_colors, "carbonne")

velocity_colors_down = [
    # (1.0, 1.0, 1.0),
    (0.0, 0.0, 0.3),  # end_val dark blue
    (0.0274509803922, 0.0549019607843, 0.23137254902),
    (0.0392156862745, 0.0549019607843, 0.250980392157),
    (0.0509803921569, 0.0549019607843, 0.266666666667),
    (0.0627450980392, 0.0549019607843, 0.286274509804),
    (0.0705882352941, 0.0549019607843, 0.298039215686),
    (0.0823529411765, 0.0549019607843, 0.313725490196),
    (0.0901960784314, 0.0549019607843, 0.333333333333),
    (0.0980392156863, 0.0549019607843, 0.349019607843),
    (0.109803921569, 0.0549019607843, 0.364705882353),
    (0.121568627451, 0.0549019607843, 0.38431372549),
    (0.133333333333, 0.0549019607843, 0.4),
    (0.141176470588, 0.0549019607843, 0.41568627451),
    (0.152941176471, 0.0549019607843, 0.43137254902),
    (0.160784313725, 0.0549019607843, 0.447058823529),
    (0.172549019608, 0.0549019607843, 0.466666666667),
    (0.18431372549, 0.0549019607843, 0.482352941176),
    (0.192156862745, 0.0549019607843, 0.498039215686),
    (0.203921568627, 0.0549019607843, 0.513725490196),
    (0.211764705882, 0.0549019607843, 0.529411764706),
    (0.223529411765, 0.0549019607843, 0.549019607843),
    (0.23137254902, 0.0549019607843, 0.564705882353),
    (0.243137254902, 0.0549019607843, 0.580392156863),
    (0.254901960784, 0.0549019607843, 0.596078431373),
    (0.262745098039, 0.0549019607843, 0.611764705882),
    (0.274509803922, 0.0549019607843, 0.627450980392),
    (0.282352941176, 0.0549019607843, 0.647058823529),
    (0.294117647059, 0.0549019607843, 0.662745098039),
    (0.305882352941, 0.0549019607843, 0.682352941176),
    (0.317647058824, 0.0549019607843, 0.698039215686),
    (0.325490196078, 0.0588235294118, 0.713725490196),
    (0.333333333333, 0.0666666666667, 0.729411764706),
    (0.345098039216, 0.105882352941, 0.745098039216),
    (0.352941176471, 0.145098039216, 0.760784313725),
    (0.364705882353, 0.18431372549, 0.780392156863),
    (0.376470588235, 0.219607843137, 0.796078431373),
    (0.388235294118, 0.258823529412, 0.811764705882),
    (0.396078431373, 0.298039215686, 0.827450980392),
    (0.403921568627, 0.337254901961, 0.843137254902),
    (0.41568627451, 0.372549019608, 0.862745098039),
    (0.427450980392, 0.411764705882, 0.878431372549),
    (0.439215686275, 0.450980392157, 0.894117647059),
    (0.447058823529, 0.486274509804, 0.909803921569),
    (0.458823529412, 0.525490196078, 0.925490196078),
    (0.466666666667, 0.564705882353, 0.945098039216),
    (0.474509803922, 0.603921568627, 0.960784313725),
    (0.486274509804, 0.639215686275, 0.976470588235),
    (0.498039215686, 0.678431372549, 0.996078431373),
    (0.482352941176, 0.678431372549, 0.956862745098),
    (0.450980392157, 0.662745098039, 0.898039215686),
    (0.419607843137, 0.647058823529, 0.835294117647),
    (0.388235294118, 0.63137254902, 0.772549019608),
    (0.356862745098, 0.61568627451, 0.713725490196),
    (0.329411764706, 0.6, 0.650980392157),
    (0.298039215686, 0.58431372549, 0.592156862745),
    (0.266666666667, 0.564705882353, 0.529411764706),
    (0.235294117647, 0.549019607843, 0.470588235294),
    (0.207843137255, 0.533333333333, 0.407843137255),
    (0.176470588235, 0.517647058824, 0.345098039216),
    (0.145098039216, 0.501960784314, 0.286274509804),
    (0.113725490196, 0.486274509804, 0.223529411765),
    (0.0823529411765, 0.470588235294, 0.164705882353),
    (0.0549019607843, 0.458823529412, 0.101960784314),
    (0.0235294117647, 0.450980392157, 0.043137254902),
    (0.00392156862745, 0.447058823529, 0.0),
    (0.0, 0.454901960784, 0.0),
    (0.0, 0.462745098039, 0.0),
    (0.0, 0.470588235294, 0.0),
    (0.0, 0.478431372549, 0.0),
    (0.0, 0.486274509804, 0.0),
    (0.0, 0.494117647059, 0.0),
    (0.0, 0.501960784314, 0.0),
    (0.0, 0.509803921569, 0.0),
    (0.0, 0.517647058824, 0.0),
    (0.0, 0.525490196078, 0.0),
    (0.0, 0.533333333333, 0.0),
    (0.0, 0.541176470588, 0.0),
    (0.0, 0.549019607843, 0.0),
    (0.0, 0.556862745098, 0.0),
    (0.0078431372549, 0.564705882353, 0.0078431372549),
    (0.0196078431373, 0.572549019608, 0.0196078431373),
    (0.0352941176471, 0.580392156863, 0.0352941176471),
    (0.0549019607843, 0.58431372549, 0.0549019607843),
    (0.078431372549, 0.588235294118, 0.078431372549),
    (0.109803921569, 0.588235294118, 0.109803921569),
    (0.129411764706, 0.6, 0.129411764706),
    (0.149019607843, 0.619607843137, 0.149019607843),
    (0.16862745098, 0.63137254902, 0.16862745098),
    (0.18431372549, 0.639215686275, 0.18431372549),
    (0.203921568627, 0.647058823529, 0.203921568627),
    (0.223529411765, 0.654901960784, 0.223529411765),
    (0.243137254902, 0.662745098039, 0.243137254902),
    (0.258823529412, 0.670588235294, 0.258823529412),
    (0.278431372549, 0.678431372549, 0.278431372549),
    (0.298039215686, 0.686274509804, 0.298039215686),
    (0.317647058824, 0.694117647059, 0.317647058824),
    (0.337254901961, 0.698039215686, 0.337254901961),
    (0.356862745098, 0.705882352941, 0.356862745098),
    (0.372549019608, 0.713725490196, 0.372549019608),
    (0.388235294118, 0.721568627451, 0.388235294118),
    (0.407843137255, 0.729411764706, 0.407843137255),
    (0.427450980392, 0.733333333333, 0.427450980392),
    (0.447058823529, 0.741176470588, 0.447058823529),
    (0.466666666667, 0.749019607843, 0.466666666667),
    (0.486274509804, 0.756862745098, 0.486274509804),
    (0.505882352941, 0.764705882353, 0.505882352941),
    (0.521568627451, 0.772549019608, 0.521568627451),
    (0.541176470588, 0.780392156863, 0.541176470588),
    (0.560784313725, 0.788235294118, 0.560784313725),
    (0.576470588235, 0.796078431373, 0.576470588235),
    (0.596078431373, 0.803921568627, 0.596078431373),
    (0.61568627451, 0.811764705882, 0.61568627451),
    (0.635294117647, 0.819607843137, 0.635294117647),
    (0.650980392157, 0.827450980392, 0.650980392157),
    (0.670588235294, 0.835294117647, 0.670588235294),
    (0.690196078431, 0.843137254902, 0.690196078431),
    (0.709803921569, 0.850980392157, 0.709803921569),
    (0.729411764706, 0.858823529412, 0.729411764706),
    (0.745098039216, 0.866666666667, 0.745098039216),
    (0.764705882353, 0.870588235294, 0.764705882353),
    (0.78431372549, 0.878431372549, 0.78431372549),
    (0.8, 0.886274509804, 0.8),                       # light green
    (0.819607843137, 0.894117647059, 0.819607843137),
    (0.839215686275, 0.901960784314, 0.839215686275),
    (0.858823529412, 0.909803921569, 0.858823529412),
    (0.878431372549, 0.917647058824, 0.878431372549),
    (0.83137254902, 0.858823529412, 0.83137254902),
    (0.737254901961, 0.745098039216, 0.737254901961),
    (0.752941176471, 0.749019607843, 0.737254901961)
]

carbonne_down_map = matplotlib.colors.ListedColormap(velocity_colors_down, "carbonne_down")

carbonne_down_r_map = matplotlib.colors.ListedColormap(velocity_colors_down[::-1], "carbonne_down_r")


ldr_colors=(
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),

(0.0 , 0.0 , 0.3),
(0.0 , 0.0 , 0.3),
(0.0 , 0.0 , 0.3),
(0.0 , 0.0 , 0.3),

(0.0 , 0.7 , 1.0),
(0.0 , 0.7 , 1.0),
(0.0 , 0.7 , 1.0),

(0.0 , 0.9 , 0.0),
(0.0 , 0.9 , 0.0),
(0.0 , 0.9 , 0.0),
(0.0 , 0.9 , 0.0),

(1.0 , 0.8 , 0.0),
(1.0 , 0.8 , 0.0),
(1.0 , 0.8 , 0.0),
(1.0 , 0.8 , 0.0),

(1.0 , 0.0 , 0.0),
(1.0 , 0.0 , 0.0),
(1.0 , 0.0 , 0.0),
(1.0 , 0.0 , 0.0),

(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6),
(0.6 , 0.6 , 0.6)
)

ldr_map = matplotlib.colors.ListedColormap(ldr_colors, "LDR")

cloudnet_colors = [[1.        , 1.        , 1.        ],
       [0.42352941, 1.        , 0.9254902 ],
       [0.1254902 , 0.62352941, 0.95294118],
       [0.74901961, 0.60392157, 1.        ],
       [0.89803922, 0.89019608, 0.92156863],
       [0.2745098 , 0.29019608, 0.7254902 ],
       [1.        , 0.64705882, 0.        ],
       [0.78039216, 0.98039216, 0.22745098],
       [0.80784314, 0.7372549 , 0.5372549 ],
       [0.90196078, 0.29019608, 0.1372549 ],
       [0.70588235, 0.21568627, 0.34117647]]
cloudnet_colors = tuple(cloudnet_colors)

cloudnet_map = matplotlib.colors.ListedColormap(cloudnet_colors, "cloudnet_target")

target_names = [
    'Clear sky',
    'Cloud liquid \ndroplets only',
    'Drizzle or rain.',
    'Drizzle/rain & \ncloud droplet',
    'Ice particles.',
    'Ice coexisting with \nsupercooled liquid \ndroplets.',
    'Melting ice particles',
    'Melting ice & \ncloud droplets',
    'Aerosol',
    'Insects',
    'Aerosol and \nInsects',
]

ldr_colors = (
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),

    (0.0, 0.0, 0.3),
    (0.0, 0.0, 0.3),
    (0.0, 0.0, 0.3),
    (0.0, 0.0, 0.3),

    (0.0, 0.7, 1.0),
    (0.0, 0.7, 1.0),
    (0.0, 0.7, 1.0),

    (0.0, 0.9, 0.0),
    (0.0, 0.9, 0.0),
    (0.0, 0.9, 0.0),
    (0.0, 0.9, 0.0),

    (1.0, 0.8, 0.0),
    (1.0, 0.8, 0.0),
    (1.0, 0.8, 0.0),
    (1.0, 0.8, 0.0),

    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),

    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6),
    (0.6, 0.6, 0.6)
)

ldr_map = matplotlib.colors.ListedColormap(ldr_colors, "LDR")
